use crate::config::IgnoreTerm;
use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use crate::translate::core;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct TranslationPolicy {
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub model_map: BTreeMap<String, String>,
    pub ignore_terms: Vec<IgnoreTerm>,
}

pub fn translate_request(
    req: anthropic::AnthropicRequest,
    policy: &TranslationPolicy,
) -> ProxyResult<openai::OpenAIRequest> {
    let model = select_model(&req, policy);

    let mut openai_messages = Vec::new();

    if let Some(system) = req.system {
        let system_texts = match system {
            anthropic::SystemPrompt::Single(text) => vec![text],
            anthropic::SystemPrompt::Multiple(messages) => {
                messages.into_iter().map(|msg| msg.text).collect()
            }
        };

        for text in system_texts {
            let sanitized = sanitize_prompt(text, &policy.ignore_terms);

            // Drop system blocks that are emptied out by term removal — forwarding an
            // empty system message is pointless and some upstreams reject it.
            if sanitized.trim().is_empty() {
                tracing::debug!("Dropping system prompt block left empty after term removal");
                continue;
            }

            openai_messages.push(openai::Message {
                role: "system".to_string(),
                content: Some(openai::MessageContent::Text(sanitized)),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    for msg in req.messages {
        openai_messages.extend(core::translate_message(msg)?);
    }

    let tools = req.tools.and_then(|tools| {
        let filtered: Vec<_> = tools
            .into_iter()
            .filter(|t| !core::is_batch_tool(t))
            .map(core::translate_tool)
            .collect();

        if filtered.is_empty() {
            None
        } else {
            Some(filtered)
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
        stream_options: req.stream.and_then(|stream| {
            stream.then_some(openai::StreamOptions {
                include_usage: true,
            })
        }),
        tools,
        tool_choice: None,
    })
}

pub fn translate_response(
    resp: openai::OpenAIResponse,
    fallback_model: &str,
) -> ProxyResult<anthropic::AnthropicResponse> {
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| ProxyError::Transform("No choices in response".to_string()))?;

    let mut content = Vec::new();

    if let Some(text) = &choice.message.content {
        if !text.is_empty() {
            content.push(anthropic::ResponseContent::Text {
                content_type: "text".to_string(),
                text: text.clone(),
            });
        }
    }

    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            let input: Value =
                serde_json::from_str(&tool_call.function.arguments).unwrap_or_else(|_| json!({}));

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
        id: resp.id.unwrap_or_else(|| "msg_proxy".to_string()),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model.unwrap_or_else(|| fallback_model.to_string()),
        stop_reason,
        stop_sequence: None,
        usage: anthropic::Usage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
        },
    })
}

pub fn translate_models_list(resp: openai::ModelsListResponse) -> anthropic::ModelsListResponse {
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

    let first_id = data.first().map(|m| m.id.clone());
    let last_id = data.last().map(|m| m.id.clone());

    anthropic::ModelsListResponse {
        data,
        first_id,
        has_more: false,
        last_id,
    }
}

fn select_model(req: &anthropic::AnthropicRequest, policy: &TranslationPolicy) -> String {
    let has_thinking = req
        .extra
        .get("thinking")
        .and_then(|v| v.as_object())
        .map(|o| o.get("type").and_then(|t| t.as_str()) == Some("enabled"))
        .unwrap_or(false);

    let model = if has_thinking {
        policy
            .reasoning_model
            .clone()
            .unwrap_or_else(|| req.model.clone())
    } else {
        policy
            .completion_model
            .clone()
            .unwrap_or_else(|| req.model.clone())
    };

    policy.model_map.get(&model).cloned().unwrap_or(model)
}

fn sanitize_prompt(text: String, terms: &[IgnoreTerm]) -> String {
    let mut sanitized = text;
    let mut removed = Vec::new();

    for term in terms {
        let next = match term {
            IgnoreTerm::Literal(literal) => core::remove_term(&sanitized, literal),
            IgnoreTerm::Regex(regex) => regex.replace_all(&sanitized, "").into_owned(),
        };
        if next != sanitized {
            sanitized = next;
            removed.push(term.describe());
        }
    }

    if !removed.is_empty() {
        tracing::debug!(
            "Removed configured system prompt terms for upstream compatibility: {}",
            removed.join("; ")
        );
    }

    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use serde_json::json;

    fn policy_from(config: &Config) -> TranslationPolicy {
        TranslationPolicy {
            reasoning_model: config.reasoning_model.clone(),
            completion_model: config.completion_model.clone(),
            model_map: config.model_map.clone(),
            ignore_terms: config.system_prompt_ignore_matchers.clone(),
        }
    }

    fn default_policy() -> TranslationPolicy {
        policy_from(&Config::default())
    }

    #[test]
    fn applies_model_map_after_selection() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("pong".to_string()),
            }],
            max_tokens: 64,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(false),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            model_map: [("claude-opus-4-6".to_string(), "openai/gpt-4.1".to_string())]
                .into_iter()
                .collect(),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "openai/gpt-4.1");
    }

    #[test]
    fn sanitizes_configured_system_prompt_terms() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("pong".to_string()),
            }],
            max_tokens: 64,
            system: Some(anthropic::SystemPrompt::Single(
                "Examples of risky actions: rm -rf.".to_string(),
            )),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(true),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            ignore_terms: vec![IgnoreTerm::Literal("rm -rf".to_string())],
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();

        match &openai.messages[0].content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "Examples of risky actions: .");
            }
            _ => panic!("expected sanitized system prompt"),
        }
    }

    #[test]
    fn sanitizes_system_prompt_with_regex_term() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("pong".to_string()),
            }],
            max_tokens: 64,
            system: Some(anthropic::SystemPrompt::Single(
                "Session id ABC-12345 then DEF-67890.".to_string(),
            )),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(true),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let terms = Config::compile_ignore_terms(&[r"/[A-Z]{3}-\d{5}/".to_string()]);
        let policy = TranslationPolicy {
            ignore_terms: terms,
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();

        match &openai.messages[0].content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "Session id  then .");
            }
            _ => panic!("expected sanitized system prompt"),
        }
    }

    #[test]
    fn drops_system_block_emptied_by_term_removal() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 64,
            system: Some(anthropic::SystemPrompt::Multiple(vec![
                anthropic::SystemMessage {
                    message_type: "text".to_string(),
                    text: "  sk-abc12345  ".to_string(),
                    cache_control: None,
                },
                anthropic::SystemMessage {
                    message_type: "text".to_string(),
                    text: "You are helpful.".to_string(),
                    cache_control: None,
                },
            ])),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let terms = Config::compile_ignore_terms(&[r"/sk-[a-z0-9]{8}/".to_string()]);
        let policy = TranslationPolicy {
            ignore_terms: terms,
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();

        let system_msgs: Vec<_> = openai
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .collect();
        assert_eq!(system_msgs.len(), 1);
        match &system_msgs[0].content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "You are helpful.");
            }
            _ => panic!("expected surviving system prompt"),
        }
    }

    #[test]
    fn streaming_request_includes_usage_stream_options() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(true),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        assert_eq!(
            openai.stream_options.map(|options| options.include_usage),
            Some(true)
        );
    }

    #[test]
    fn non_streaming_request_omits_usage_stream_options() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(false),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        assert!(openai.stream_options.is_none());
    }

    #[test]
    fn converts_tool_definitions() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("use tool".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: Some(vec![anthropic::Tool {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
                tool_type: None,
            }]),
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        let tools = openai.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "read_file");
    }

    #[test]
    fn filters_batch_tools() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: Some(vec![anthropic::Tool {
                name: "batch_tool".to_string(),
                description: None,
                input_schema: json!({}),
                tool_type: Some("BatchTool".to_string()),
            }]),
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();
        assert!(openai.tools.is_none());
    }

    #[test]
    fn converts_image_content() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Blocks(vec![
                    anthropic::ContentBlock::Text {
                        text: "What is this?".to_string(),
                        cache_control: None,
                    },
                    anthropic::ContentBlock::Image {
                        source: anthropic::ImageSource {
                            source_type: "base64".to_string(),
                            media_type: "image/png".to_string(),
                            data: "iVBOR...".to_string(),
                        },
                    },
                ]),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        match &openai.messages[0].content {
            Some(openai::MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                match &parts[1] {
                    openai::ContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/png;base64,"));
                    }
                    _ => panic!("expected image_url part"),
                }
            }
            _ => panic!("expected multi-part content"),
        }
    }

    #[test]
    fn converts_tool_use_and_tool_result() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                anthropic::Message {
                    role: "assistant".to_string(),
                    content: anthropic::MessageContent::Blocks(vec![
                        anthropic::ContentBlock::ToolUse {
                            id: "tool_1".to_string(),
                            name: "read_file".to_string(),
                            input: json!({"path": "/tmp"}),
                        },
                    ]),
                },
                anthropic::Message {
                    role: "user".to_string(),
                    content: anthropic::MessageContent::Blocks(vec![
                        anthropic::ContentBlock::ToolResult {
                            tool_use_id: "tool_1".to_string(),
                            content: anthropic::ToolResultContent::Text(
                                "file contents".to_string(),
                            ),
                            is_error: None,
                        },
                    ]),
                },
            ],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        let tool_calls = openai.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].id, "tool_1");
        assert_eq!(tool_calls[0].function.name, "read_file");

        assert_eq!(openai.messages[1].role, "tool");
        assert_eq!(openai.messages[1].tool_call_id, Some("tool_1".to_string()));
    }

    #[test]
    fn deserializes_tool_result_with_nested_content_blocks() {
        let body = json!({
            "model": "gpt-4o",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool_42",
                    "content": [
                        {"type": "text", "text": "first chunk"},
                        {"type": "text", "text": "second chunk"}
                    ]
                }]
            }]
        });

        let req: anthropic::AnthropicRequest = serde_json::from_value(body).unwrap();
        let openai = translate_request(req, &default_policy()).unwrap();

        let tool_msg = openai
            .messages
            .iter()
            .find(|m| m.role == "tool")
            .expect("expected a tool message");
        assert_eq!(tool_msg.tool_call_id, Some("tool_42".to_string()));
        match &tool_msg.content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "first chunk\nsecond chunk");
            }
            other => panic!("expected flattened text content, got {:?}", other),
        }
    }

    #[test]
    fn converts_multiple_system_prompts() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: Some(anthropic::SystemPrompt::Multiple(vec![
                anthropic::SystemMessage {
                    message_type: "text".to_string(),
                    text: "You are helpful.".to_string(),
                    cache_control: None,
                },
                anthropic::SystemMessage {
                    message_type: "text".to_string(),
                    text: "Be concise.".to_string(),
                    cache_control: None,
                },
            ])),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        let system_msgs: Vec<_> = openai
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .collect();
        assert_eq!(system_msgs.len(), 2);
    }

    #[test]
    fn uses_reasoning_model_when_thinking_enabled() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("think hard".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({"thinking": {"type": "enabled", "budget_tokens": 1000}}),
        };

        let policy = TranslationPolicy {
            reasoning_model: Some("gpt-4o-reasoning".to_string()),
            completion_model: Some("gpt-4o-mini".to_string()),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "gpt-4o-reasoning");
    }

    #[test]
    fn uses_completion_model_without_thinking() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("quick".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            reasoning_model: Some("gpt-4o-reasoning".to_string()),
            completion_model: Some("gpt-4o-mini".to_string()),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "gpt-4o-mini");
    }

    #[test]
    fn response_with_all_fields_present() {
        let response = openai::OpenAIResponse {
            id: Some("chatcmpl-abc123".to_string()),
            object: Some("chat.completion".to_string()),
            created: Some(1700000000),
            model: Some("gpt-4o".to_string()),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some("hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "fallback-model").unwrap();
        assert_eq!(anthropic.id, "chatcmpl-abc123");
        assert_eq!(anthropic.model, "gpt-4o");
    }

    #[test]
    fn response_allows_missing_metadata() {
        let response = openai::OpenAIResponse {
            id: None,
            object: None,
            created: None,
            model: None,
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some("pong".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "openai/gpt-4o-mini").unwrap();
        assert_eq!(anthropic.id, "msg_proxy");
        assert_eq!(anthropic.model, "openai/gpt-4o-mini");
    }

    #[test]
    fn response_converts_tool_calls() {
        let response = openai::OpenAIResponse {
            id: Some("chatcmpl-1".to_string()),
            object: None,
            created: None,
            model: Some("gpt-4o".to_string()),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![openai::ToolCall {
                        id: "call_abc".to_string(),
                        call_type: "function".to_string(),
                        function: openai::FunctionCall {
                            name: "read_file".to_string(),
                            arguments: "{\"path\":\"/tmp\"}".to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "fallback").unwrap();
        assert_eq!(anthropic.stop_reason, Some("tool_use".to_string()));
        assert!(!anthropic.content.is_empty());
    }

    #[test]
    fn models_list_translation() {
        let response = openai::ModelsListResponse {
            object: Some("list".to_string()),
            data: vec![
                openai::ModelInfo {
                    id: "gpt-4o-mini".to_string(),
                    object: Some("model".to_string()),
                    created: None,
                    owned_by: Some("azure".to_string()),
                },
                openai::ModelInfo {
                    id: "gpt-5-chat".to_string(),
                    object: Some("model".to_string()),
                    created: None,
                    owned_by: Some("azure".to_string()),
                },
            ],
        };

        let result = translate_models_list(response);
        assert_eq!(result.first_id.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(result.last_id.as_deref(), Some("gpt-5-chat"));
        assert!(!result.has_more);
    }

    #[test]
    fn empty_models_list() {
        let response = openai::ModelsListResponse {
            object: Some("list".to_string()),
            data: vec![],
        };
        let result = translate_models_list(response);
        assert!(result.data.is_empty());
        assert!(result.first_id.is_none());
    }
}
