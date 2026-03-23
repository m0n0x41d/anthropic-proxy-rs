use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use serde_json::{json, Value};

/// Transform Anthropic request to OpenAI format
pub fn anthropic_to_openai(
    req: anthropic::AnthropicRequest,
    config: &Config,
) -> ProxyResult<openai::OpenAIRequest> {
    // Determine model based on thinking parameter
    let has_thinking = req
        .extra
        .get("thinking")
        .and_then(|v| v.as_object())
        .map(|o| o.get("type").and_then(|t| t.as_str()) == Some("enabled"))
        .unwrap_or(false);

    // Use configured model or fall back to the model from the request
    let model = if has_thinking {
        config.reasoning_model.clone()
            .or_else(|| Some(req.model.clone()))
            .unwrap_or_else(|| req.model.clone())
    } else {
        config.completion_model.clone()
            .or_else(|| Some(req.model.clone()))
            .unwrap_or_else(|| req.model.clone())
    };

    // Convert messages
    let mut openai_messages = Vec::new();

    // Add system message if present
    if let Some(system) = req.system {
        match system {
            anthropic::SystemPrompt::Single(text) => {
                openai_messages.push(openai::Message {
                    role: "system".to_string(),
                    content: Some(openai::MessageContent::Text(text)),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            anthropic::SystemPrompt::Multiple(messages) => {
                for msg in messages {
                    openai_messages.push(openai::Message {
                        role: "system".to_string(),
                        content: Some(openai::MessageContent::Text(msg.text)),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
        }
    }

    // Convert user/assistant messages
    for msg in req.messages {
        let converted = convert_message(msg)?;
        openai_messages.extend(converted);
    }

    // Convert tools
    let tools = req.tools.and_then(|tools| {
        let filtered: Vec<_> = tools
            .into_iter()
            .filter(|t| t.tool_type.as_deref() != Some("BatchTool"))
            .collect();

        if filtered.is_empty() {
            None
        } else {
            Some(
                filtered
                    .into_iter()
                    .map(|t| openai::Tool {
                        tool_type: "function".to_string(),
                        function: openai::Function {
                            name: t.name,
                            description: t.description,
                            parameters: clean_schema(t.input_schema),
                        },
                    })
                    .collect(),
            )
        }
    });

    Ok(openai::OpenAIRequest {
        model,
        messages: openai_messages,
        max_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences,
        stream: req.stream,
        tools,
        tool_choice: None,
    })
}

/// Convert a single Anthropic message to one or more OpenAI messages
fn convert_message(msg: anthropic::Message) -> ProxyResult<Vec<openai::Message>> {
    let mut result = Vec::new();

    match msg.content {
        anthropic::MessageContent::Text(text) => {
            result.push(openai::Message {
                role: msg.role,
                content: Some(openai::MessageContent::Text(text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        anthropic::MessageContent::Blocks(blocks) => {
            let mut current_content_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for block in blocks {
                match block {
                    anthropic::ContentBlock::Text { text, .. } => {
                        current_content_parts.push(openai::ContentPart::Text { text });
                    }
                    anthropic::ContentBlock::Image { source } => {
                        let data_url = format!(
                            "data:{};base64,{}",
                            source.media_type, source.data
                        );
                        current_content_parts.push(openai::ContentPart::ImageUrl {
                            image_url: openai::ImageUrl { url: data_url },
                        });
                    }
                    anthropic::ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(openai::ToolCall {
                            id,
                            call_type: "function".to_string(),
                            function: openai::FunctionCall {
                                name,
                                arguments: serde_json::to_string(&input)
                                    .map_err(|e| ProxyError::Serialization(e))?,
                            },
                        });
                    }
                    anthropic::ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        // Tool results become separate messages with role "tool"
                        result.push(openai::Message {
                            role: "tool".to_string(),
                            content: Some(openai::MessageContent::Text(content)),
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id),
                            name: None,
                        });
                    }
                    anthropic::ContentBlock::Thinking { .. } => {
                        // Skip thinking blocks in request
                    }
                }
            }

            // Add message with content and/or tool calls
            if !current_content_parts.is_empty() || !tool_calls.is_empty() {
                let content = if current_content_parts.is_empty() {
                    None
                } else if current_content_parts.len() == 1 {
                    match &current_content_parts[0] {
                        openai::ContentPart::Text { text } => {
                            Some(openai::MessageContent::Text(text.clone()))
                        }
                        _ => Some(openai::MessageContent::Parts(current_content_parts)),
                    }
                } else {
                    Some(openai::MessageContent::Parts(current_content_parts))
                };

                result.push(openai::Message {
                    role: msg.role,
                    content,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                    name: None,
                });
            }
        }
    }

    Ok(result)
}

/// Clean JSON schema by removing unsupported formats
fn clean_schema(mut schema: Value) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        // Remove "format": "uri"
        if obj.get("format").and_then(|v| v.as_str()) == Some("uri") {
            obj.remove("format");
        }

        // Recursively clean nested schemas
        if let Some(properties) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
            for (_, value) in properties.iter_mut() {
                *value = clean_schema(value.clone());
            }
        }

        if let Some(items) = obj.get_mut("items") {
            *items = clean_schema(items.clone());
        }
    }

    schema
}

/// Transform OpenAI response to Anthropic format
pub fn openai_to_anthropic(
    resp: openai::OpenAIResponse,
) -> ProxyResult<anthropic::AnthropicResponse> {
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| ProxyError::Transform("No choices in response".to_string()))?;

    let mut content = Vec::new();

    // Add text content if present
    if let Some(text) = &choice.message.content {
        if !text.is_empty() {
            content.push(anthropic::ResponseContent::Text {
                content_type: "text".to_string(),
                text: text.clone(),
            });
        }
    }

    // Add tool calls if present
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            let input: Value = serde_json::from_str(&tool_call.function.arguments)
                .unwrap_or_else(|_| json!({}));

            content.push(anthropic::ResponseContent::ToolUse {
                content_type: "tool_use".to_string(),
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
            });
        }
    }

    let stop_reason = choice
        .finish_reason
        .as_ref()
        .map(|r| match r.as_str() {
            "tool_calls" => "tool_use",
            "stop" => "end_turn",
            "length" => "max_tokens",
            _ => "end_turn",
        })
        .map(String::from);

    Ok(anthropic::AnthropicResponse {
        id: resp.id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model,
        stop_reason,
        stop_sequence: None,
        usage: anthropic::Usage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
        },
    })
}

pub fn openai_models_to_anthropic(
    resp: openai::ModelsListResponse,
) -> anthropic::ModelsListResponse {
    let data: Vec<_> = resp
        .data
        .into_iter()
        .map(|model| anthropic::ModelInfo {
            created_at: "1970-01-01T00:00:00Z".to_string(),
            display_name: model.id.clone(),
            id: model.id,
            model_type: "model".to_string(),
        })
        .collect();

    let first_id = data.first().map(|model| model.id.clone());
    let last_id = data.last().map(|model| model.id.clone());

    anthropic::ModelsListResponse {
        data,
        first_id,
        has_more: false,
        last_id,
    }
}

/// Map OpenAI finish reason to Anthropic stop reason
pub fn map_stop_reason(finish_reason: Option<&str>) -> Option<String> {
    finish_reason.map(|r| match r {
        "tool_calls" => "tool_use",
        "stop" => "end_turn",
        "length" => "max_tokens",
        _ => "end_turn",
    }.to_string())
}

#[cfg(test)]
mod tests {
    use crate::models::openai;

    #[test]
    fn openai_models_are_mapped_to_anthropic_shape() {
        let response = openai::ModelsListResponse {
            object: Some("list".to_string()),
            data: vec![
                openai::ModelInfo {
                    id: "openai/gpt-4o-mini".to_string(),
                    object: Some("model".to_string()),
                    created: None,
                    owned_by: Some("azure".to_string()),
                },
                openai::ModelInfo {
                    id: "openai/gpt-5-chat".to_string(),
                    object: Some("model".to_string()),
                    created: None,
                    owned_by: Some("azure".to_string()),
                },
            ],
        };

        let anthropic = super::openai_models_to_anthropic(response);

        assert_eq!(anthropic.first_id.as_deref(), Some("openai/gpt-4o-mini"));
        assert_eq!(anthropic.last_id.as_deref(), Some("openai/gpt-5-chat"));
        assert!(!anthropic.has_more);
        assert_eq!(anthropic.data[0].model_type, "model");
    }

    #[test]
    fn empty_models_list_produces_empty_response() {
        let response = openai::ModelsListResponse {
            object: Some("list".to_string()),
            data: vec![],
        };

        let anthropic = super::openai_models_to_anthropic(response);

        assert!(anthropic.data.is_empty());
        assert!(anthropic.first_id.is_none());
        assert!(anthropic.last_id.is_none());
        assert!(!anthropic.has_more);
    }

    #[test]
    fn single_model_has_same_first_and_last_id() {
        let response = openai::ModelsListResponse {
            object: None,
            data: vec![openai::ModelInfo {
                id: "only-model".to_string(),
                object: None,
                created: None,
                owned_by: None,
            }],
        };

        let anthropic = super::openai_models_to_anthropic(response);

        assert_eq!(anthropic.first_id.as_deref(), Some("only-model"));
        assert_eq!(anthropic.last_id.as_deref(), Some("only-model"));
        assert_eq!(anthropic.data.len(), 1);
        assert_eq!(anthropic.data[0].display_name, "only-model");
    }
}
