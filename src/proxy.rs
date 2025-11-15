use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use crate::transform;
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
    Extension, Json,
};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

pub async fn proxy_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    Json(req): Json<anthropic::AnthropicRequest>,
) -> ProxyResult<Response> {
    let is_streaming = req.stream.unwrap_or(false);

    tracing::debug!("Received request for model: {}", req.model);
    tracing::debug!("Streaming: {}", is_streaming);

    if config.verbose {
        tracing::trace!("Incoming Anthropic request: {}", serde_json::to_string_pretty(&req).unwrap_or_default());
    }

    let openai_req = transform::anthropic_to_openai(req, &config)?;

    if config.verbose {
        tracing::trace!("Transformed OpenAI request: {}", serde_json::to_string_pretty(&openai_req).unwrap_or_default());
    }

    if is_streaming {
        handle_streaming(config, client, openai_req).await
    } else {
        handle_non_streaming(config, client, openai_req).await
    }
}

async fn handle_non_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
) -> ProxyResult<Response> {
    let url = config.chat_completions_url();
    tracing::debug!("Sending non-streaming request to {}", url);
    tracing::debug!("Request model: {}", openai_req.model);

    let mut req_builder = client
        .post(&config.chat_completions_url())
        .json(&openai_req)
        .timeout(Duration::from_secs(300));

    if let Some(api_key) = &config.api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        tracing::error!("Upstream error ({}): {}", status, error_text);
        return Err(ProxyError::Upstream(format!(
            "Upstream returned {}: {}",
            status, error_text
        )));
    }

    let openai_resp: openai::OpenAIResponse = response.json().await?;

    if config.verbose {
        tracing::trace!("Received OpenAI response: {}", serde_json::to_string_pretty(&openai_resp).unwrap_or_default());
    }

    let anthropic_resp = transform::openai_to_anthropic(openai_resp)?;

    if config.verbose {
        tracing::trace!("Transformed Anthropic response: {}", serde_json::to_string_pretty(&anthropic_resp).unwrap_or_default());
    }

    Ok(Json(anthropic_resp).into_response())
}

async fn handle_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
) -> ProxyResult<Response> {
    let url = config.chat_completions_url();
    tracing::debug!("Sending streaming request to {}", url);
    tracing::debug!("Request model: {}", openai_req.model);

    let mut req_builder = client
        .post(&config.chat_completions_url())
        .json(&openai_req)
        .timeout(Duration::from_secs(300));

    if let Some(api_key) = &config.api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = req_builder.send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        tracing::error!(
            "Upstream error ({}) from {}: {}", 
            status,
            url,
            error_text
        );
        return Err(ProxyError::Upstream(format!(
            "Upstream returned {} from {}: {}",
            status, url, error_text
        )));
    }

    let stream = response.bytes_stream();
    let sse_stream = create_sse_stream(stream);

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    headers.insert("Connection", HeaderValue::from_static("keep-alive"));

    Ok((headers, Body::from_stream(sse_stream)).into_response())
}

fn create_sse_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut message_id = None;
        let mut current_model = None;
        let mut content_index = 0;
        let mut tool_call_id = None;
        let mut _tool_call_name = None;
        let mut tool_call_args = String::new();
        let mut has_sent_message_start = false;
        let mut current_block_type: Option<String> = None;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    while let Some(pos) = buffer.find("\n\n") {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if line.trim().is_empty() {
                            continue;
                        }

                        for l in line.lines() {
                            if let Some(data) = l.strip_prefix("data: ") {
                                if data.trim() == "[DONE]" {
                                    let event = json!({"type": "message_stop"});
                                    let sse_data = format!("event: message_stop\ndata: {}\n\n",
                                        serde_json::to_string(&event).unwrap_or_default());
                                    yield Ok(Bytes::from(sse_data));
                                    continue;
                                }

                                if let Ok(chunk) = serde_json::from_str::<openai::StreamChunk>(data) {
                                    if message_id.is_none() {
                                        message_id = Some(chunk.id.clone());
                                    }
                                    if current_model.is_none() {
                                        current_model = Some(chunk.model.clone());
                                    }

                                    if let Some(choice) = chunk.choices.first() {
                                        if !has_sent_message_start {
                                            let event = anthropic::StreamEvent::MessageStart {
                                                message: anthropic::MessageStartData {
                                                    id: message_id.clone().unwrap_or_default(),
                                                    message_type: "message".to_string(),
                                                    role: "assistant".to_string(),
                                                    model: current_model.clone().unwrap_or_default(),
                                                    usage: anthropic::Usage {
                                                        input_tokens: 0,
                                                        output_tokens: 0,
                                                    },
                                                },
                                            };
                                            let sse_data = format!("event: message_start\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                            has_sent_message_start = true;
                                        }

                                        if let Some(reasoning) = &choice.delta.reasoning {
                                            if current_block_type.is_none() {
                                                let event = json!({
                                                    "type": "content_block_start",
                                                    "index": content_index,
                                                    "content_block": {
                                                        "type": "thinking",
                                                        "thinking": ""
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                                current_block_type = Some("thinking".to_string());
                                            }

                                            let event = json!({
                                                "type": "content_block_delta",
                                                "index": content_index,
                                                "delta": {
                                                    "type": "thinking_delta",
                                                    "thinking": reasoning
                                                }
                                            });
                                            let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                        }

                                        if let Some(content) = &choice.delta.content {
                                            if !content.is_empty() {
                                                if current_block_type.as_deref() != Some("text") {
                                                    if current_block_type.is_some() {
                                                        let event = json!({
                                                            "type": "content_block_stop",
                                                            "index": content_index
                                                        });
                                                        let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        content_index += 1;
                                                    }

                                                    // Start text block
                                                    let event = json!({
                                                        "type": "content_block_start",
                                                        "index": content_index,
                                                        "content_block": {
                                                            "type": "text",
                                                            "text": ""
                                                        }
                                                    });
                                                    let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default());
                                                    yield Ok(Bytes::from(sse_data));
                                                    current_block_type = Some("text".to_string());
                                                }

                                                // Send text delta
                                                let event = json!({
                                                    "type": "content_block_delta",
                                                    "index": content_index,
                                                    "delta": {
                                                        "type": "text_delta",
                                                        "text": content
                                                    }
                                                });
                                                let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                            }
                                        }

                                        // Handle tool calls
                                        if let Some(tool_calls) = &choice.delta.tool_calls {
                                            for tool_call in tool_calls {
                                                if let Some(id) = &tool_call.id {
                                                    // Start of new tool call
                                                    if current_block_type.is_some() {
                                                        let event = json!({
                                                            "type": "content_block_stop",
                                                            "index": content_index
                                                        });
                                                        let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        content_index += 1;
                                                    }

                                                    tool_call_id = Some(id.clone());
                                                    tool_call_args.clear();
                                                }

                                                if let Some(function) = &tool_call.function {
                                                    if let Some(name) = &function.name {
                                                        _tool_call_name = Some(name.clone());

                                                        // Start tool_use block
                                                        let event = json!({
                                                            "type": "content_block_start",
                                                            "index": content_index,
                                                            "content_block": {
                                                                "type": "tool_use",
                                                                "id": tool_call_id.clone().unwrap_or_default(),
                                                                "name": name
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_start\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                        current_block_type = Some("tool_use".to_string());
                                                    }

                                                    if let Some(args) = &function.arguments {
                                                        tool_call_args.push_str(args);

                                                        // Send input_json_delta
                                                        let event = json!({
                                                            "type": "content_block_delta",
                                                            "index": content_index,
                                                            "delta": {
                                                                "type": "input_json_delta",
                                                                "partial_json": args
                                                            }
                                                        });
                                                        let sse_data = format!("event: content_block_delta\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default());
                                                        yield Ok(Bytes::from(sse_data));
                                                    }
                                                }
                                            }
                                        }

                                        // Handle finish reason
                                        if let Some(finish_reason) = &choice.finish_reason {
                                            // Close current content block
                                            if current_block_type.is_some() {
                                                let event = json!({
                                                    "type": "content_block_stop",
                                                    "index": content_index
                                                });
                                                let sse_data = format!("event: content_block_stop\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default());
                                                yield Ok(Bytes::from(sse_data));
                                            }

                                            // Send message_delta with stop_reason
                                            let stop_reason = transform::map_stop_reason(Some(finish_reason));
                                            let event = json!({
                                                "type": "message_delta",
                                                "delta": {
                                                    "stop_reason": stop_reason,
                                                    "stop_sequence": serde_json::Value::Null
                                                },
                                                "usage": chunk.usage.as_ref().map(|u| json!({
                                                    "output_tokens": u.completion_tokens
                                                }))
                                            });
                                            let sse_data = format!("event: message_delta\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default());
                                            yield Ok(Bytes::from(sse_data));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    let error_event = json!({
                        "type": "error",
                        "error": {
                            "type": "stream_error",
                            "message": format!("Stream error: {}", e)
                        }
                    });
                    let sse_data = format!("event: error\ndata: {}\n\n",
                        serde_json::to_string(&error_event).unwrap_or_default());
                    yield Ok(Bytes::from(sse_data));
                    break;
                }
            }
        }
    }
}
