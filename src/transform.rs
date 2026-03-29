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
                    content: Some(openai::MessageContent::Text(sanitize_system_prompt(
                        text, config,
                    ))),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            anthropic::SystemPrompt::Multiple(messages) => {
                for msg in messages {
                    openai_messages.push(openai::Message {
                        role: "system".to_string(),
                        content: Some(openai::MessageContent::Text(sanitize_system_prompt(
                            msg.text, config,
                        ))),
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

fn sanitize_system_prompt(text: String, config: &Config) -> String {
    let mut sanitized = text;
    let mut removed_terms = Vec::new();

    for term in &config.system_prompt_ignore_terms {
        let next = remove_ignored_term(&sanitized, term);
        if next != sanitized {
            sanitized = next;
            removed_terms.push(term.clone());
        }
    }

    if !removed_terms.is_empty() {
        tracing::debug!(
            "Removed configured system prompt terms for upstream compatibility: {}",
            removed_terms.join("; ")
        );
    }

    sanitized
}

fn remove_ignored_term(text: &str, term: &str) -> String {
    let tokens: Vec<Vec<u8>> = term
        .split_whitespace()
        .map(|token| {
            token
                .as_bytes()
                .iter()
                .map(u8::to_ascii_lowercase)
                .collect()
        })
        .collect();

    if tokens.is_empty() {
        return text.to_string();
    }

    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        if let Some(end) = match_ignore_term_at(bytes, index, &tokens) {
            spans.push((index, end));
            index = end;
        } else {
            index += 1;
        }
    }

    if spans.is_empty() {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut cursor = 0;

    for (start, end) in spans {
        result.push_str(&text[cursor..start]);
        cursor = end;
    }

    result.push_str(&text[cursor..]);
    result
}

fn match_ignore_term_at(text: &[u8], start: usize, tokens: &[Vec<u8>]) -> Option<usize> {
    let mut index = start;

    if requires_boundary(text.get(start).copied())
        && is_boundary_char(text.get(start.wrapping_sub(1)).copied())
    {
        return None;
    }

    for (token_index, token) in tokens.iter().enumerate() {
        if token_index > 0 {
            let whitespace_start = index;
            while index < text.len() && text[index].is_ascii_whitespace() {
                index += 1;
            }
            if whitespace_start == index {
                return None;
            }
        }

        for expected in token {
            if index >= text.len() || text[index].to_ascii_lowercase() != *expected {
                return None;
            }
            index += 1;
        }
    }

    if requires_boundary(text.get(index.saturating_sub(1)).copied())
        && is_boundary_char(text.get(index).copied())
    {
        return None;
    }

    Some(index)
}

fn requires_boundary(byte: Option<u8>) -> bool {
    byte.is_some_and(is_boundary_byte)
}

fn is_boundary_char(byte: Option<u8>) -> bool {
    byte.is_some_and(is_boundary_byte)
}

fn is_boundary_byte(value: u8) -> bool {
    value.is_ascii_alphanumeric() || value == b'_'
}

/// Clean JSON schema by removing unsupported formats
fn clean_schema(schema: Value) -> Value {
    match schema {
        Value::Object(mut obj) => {
            obj.retain(|_, value| !value.is_null());

            if obj.get("format").and_then(|v| v.as_str()) == Some("uri") {
                obj.remove("format");
            }

            if let Some(properties) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
                for (_, value) in properties.iter_mut() {
                    *value = clean_schema(value.clone());
                }
            }

            for key in [
                "items",
                "additionalProperties",
                "contains",
                "not",
                "if",
                "then",
                "else",
            ] {
                if let Some(value) = obj.get_mut(key) {
                    *value = clean_schema(value.clone());
                }
            }

            for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
                if let Some(values) = obj.get_mut(key).and_then(|v| v.as_array_mut()) {
                    for value in values.iter_mut() {
                        *value = clean_schema(value.clone());
                    }
                }
            }

            if obj.get("type").and_then(|v| v.as_str()) == Some("object")
                && !obj.contains_key("required")
            {
                obj.insert("required".to_string(), Value::Array(Vec::new()));
            }

            if let Some(required) = obj.get_mut("required") {
                if !required.is_array() {
                    *required = Value::Array(Vec::new());
                }
            }

            Value::Object(obj)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(clean_schema).collect()),
        other => other,
    }
}

/// Transform OpenAI response to Anthropic format
pub fn openai_to_anthropic(
    resp: openai::OpenAIResponse,
    fallback_model: &str,
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
    use super::{clean_schema, openai_to_anthropic, remove_ignored_term, sanitize_system_prompt};
    use crate::config::Config;
    use crate::models::{anthropic, openai};
    use serde_json::json;

    #[test]
    fn openai_response_allows_missing_metadata_fields() {
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

        let anthropic = openai_to_anthropic(response, "openai/gpt-4o-mini").unwrap();

        assert_eq!(anthropic.id, "msg_proxy");
        assert_eq!(anthropic.model, "openai/gpt-4o-mini");
        assert_eq!(anthropic.usage.input_tokens, 10);
        assert_eq!(anthropic.usage.output_tokens, 2);
    }

    #[test]
    fn openai_response_with_all_fields_present_uses_them() {
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

        let anthropic = openai_to_anthropic(response, "fallback-model").unwrap();

        assert_eq!(anthropic.id, "chatcmpl-abc123");
        assert_eq!(anthropic.model, "gpt-4o");
    }

    #[test]
    fn clean_schema_adds_empty_required_to_object_schemas() {
        let schema = json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "format": "uri"
                }
            }
        });

        let cleaned = clean_schema(schema);

        assert_eq!(cleaned["required"], json!([]));
        assert!(cleaned["properties"]["prompt"].get("format").is_none());
    }

    #[test]
    fn clean_schema_normalizes_non_array_required() {
        let schema = json!({
            "type": "object",
            "required": null
        });

        let cleaned = clean_schema(schema);

        assert_eq!(cleaned["required"], json!([]));
    }

    #[test]
    fn remove_ignored_term_matches_rm_rf_case_insensitively_with_flexible_whitespace() {
        let sanitized =
            remove_ignored_term("Avoid destructive operations such as RM\t-rF.", "rm -rf");

        assert_eq!(sanitized, "Avoid destructive operations such as .");
    }

    #[test]
    fn remove_ignored_term_respects_word_boundaries() {
        let sanitized = remove_ignored_term("farm -rf should not match rm -rf", "rm -rf");

        assert_eq!(sanitized, "farm -rf should not match ");
    }

    #[test]
    fn sanitize_system_prompt_removes_configured_terms() {
        let config = Config {
            port: 3000,
            base_url: "https://example.com".to_string(),
            api_key: None,
            system_prompt_ignore_terms: vec!["rm -rf".to_string()],
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
        };

        let sanitized =
            sanitize_system_prompt("Examples of risky actions: rm -rf.".to_string(), &config);

        assert_eq!(sanitized, "Examples of risky actions: .");
    }

    #[test]
    fn anthropic_to_openai_sanitizes_configured_system_prompt_terms() {
        let req = anthropic::AnthropicRequest {
            model: "openai/gpt-4.1-mini".to_string(),
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

        let config = Config {
            port: 3000,
            base_url: "https://example.com".to_string(),
            api_key: None,
            system_prompt_ignore_terms: vec!["rm -rf".to_string()],
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
        };

        let openai = super::anthropic_to_openai(req, &config).unwrap();

        match &openai.messages[0].content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "Examples of risky actions: .");
            }
            _ => panic!("expected sanitized system prompt"),
        }
    }
}
