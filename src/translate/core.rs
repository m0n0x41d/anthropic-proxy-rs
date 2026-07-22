use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use serde_json::Value;

pub fn translate_message(msg: anthropic::Message) -> ProxyResult<Vec<openai::Message>> {
    let mut result = Vec::new();

    match msg.content {
        anthropic::MessageContent::Text(text) => {
            result.push(openai::Message {
                role: msg.role,
                content: Some(openai::MessageContent::Text(text)),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        anthropic::MessageContent::Blocks(blocks) => {
            let mut content_parts = Vec::new();
            let mut reasoning_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for block in blocks {
                match block {
                    anthropic::ContentBlock::Text { text, .. } => {
                        content_parts.push(openai::ContentPart::Text { text });
                    }
                    anthropic::ContentBlock::Image { source } => {
                        let data_url = format!("data:{};base64,{}", source.media_type, source.data);
                        content_parts.push(openai::ContentPart::ImageUrl {
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
                                    .map_err(ProxyError::Serialization)?,
                            },
                        });
                    }
                    anthropic::ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        let text = match content {
                            anthropic::ToolResultContent::Text(s) => s,
                            anthropic::ToolResultContent::Blocks(blocks) => blocks
                                .into_iter()
                                .filter_map(|b| match b {
                                    anthropic::ContentBlock::Text { text, .. } => Some(text),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        result.push(openai::Message {
                            role: "tool".to_string(),
                            content: Some(openai::MessageContent::Text(text)),
                            reasoning_content: None,
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id),
                            name: None,
                        });
                    }
                    anthropic::ContentBlock::Thinking { thinking } => {
                        if !thinking.is_empty() {
                            reasoning_parts.push(thinking);
                        }
                    }
                    anthropic::ContentBlock::Unknown => {
                        tracing::warn!(
                            "dropping unsupported content block before forwarding upstream"
                        );
                    }
                }
            }

            if !content_parts.is_empty() || !tool_calls.is_empty() || !reasoning_parts.is_empty() {
                let content = if content_parts.is_empty() {
                    None
                } else if content_parts.len() == 1 {
                    match &content_parts[0] {
                        openai::ContentPart::Text { text } => {
                            Some(openai::MessageContent::Text(text.clone()))
                        }
                        _ => Some(openai::MessageContent::Parts(content_parts)),
                    }
                } else {
                    Some(openai::MessageContent::Parts(content_parts))
                };

                result.push(openai::Message {
                    role: msg.role,
                    content,
                    reasoning_content: if reasoning_parts.is_empty() {
                        None
                    } else {
                        Some(reasoning_parts.join(""))
                    },
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

pub fn translate_tool(tool: anthropic::Tool) -> openai::Tool {
    openai::Tool {
        tool_type: "function".to_string(),
        function: openai::Function {
            name: tool.name,
            description: tool.description,
            parameters: normalize_schema(tool.input_schema),
        },
    }
}

pub fn is_batch_tool(tool: &anthropic::Tool) -> bool {
    tool.tool_type.as_deref() == Some("BatchTool")
}

pub fn normalize_schema(schema: Value) -> Value {
    match schema {
        Value::Object(mut obj) => {
            obj.retain(|_, value| !value.is_null());

            if obj.get("format").and_then(|v| v.as_str()) == Some("uri") {
                obj.remove("format");
            }

            if let Some(properties) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
                for (_, value) in properties.iter_mut() {
                    *value = normalize_schema(value.clone());
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
                    *value = normalize_schema(value.clone());
                }
            }

            for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
                if let Some(values) = obj.get_mut(key).and_then(|v| v.as_array_mut()) {
                    for value in values.iter_mut() {
                        *value = normalize_schema(value.clone());
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
        Value::Array(values) => Value::Array(values.into_iter().map(normalize_schema).collect()),
        other => other,
    }
}

pub fn remove_term(text: &str, term: &str) -> String {
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
        if let Some(end) = match_term_at(bytes, index, &tokens) {
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

pub fn map_stop_reason(finish_reason: Option<&str>) -> Option<String> {
    finish_reason.map(|r| {
        match r {
            "tool_calls" => "tool_use",
            "stop" => "end_turn",
            "length" => "max_tokens",
            _ => "end_turn",
        }
        .to_string()
    })
}

fn match_term_at(text: &[u8], start: usize, tokens: &[Vec<u8>]) -> Option<usize> {
    let mut index = start;

    if is_word_byte(text.get(start).copied())
        && is_word_byte(text.get(start.wrapping_sub(1)).copied())
    {
        return None;
    }

    for (token_index, token) in tokens.iter().enumerate() {
        if token_index > 0 {
            let ws_start = index;
            while index < text.len() && text[index].is_ascii_whitespace() {
                index += 1;
            }
            if ws_start == index {
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

    if is_word_byte(text.get(index.saturating_sub(1)).copied())
        && is_word_byte(text.get(index).copied())
    {
        return None;
    }

    Some(index)
}

fn is_word_byte(byte: Option<u8>) -> bool {
    byte.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn thinking_block_becomes_reasoning_content() {
        let msg = anthropic::Message {
            role: "assistant".to_string(),
            content: anthropic::MessageContent::Blocks(vec![
                anthropic::ContentBlock::Thinking {
                    thinking: "I should preserve this".to_string(),
                },
                anthropic::ContentBlock::Text {
                    text: "Answer".to_string(),
                    cache_control: None,
                },
            ]),
        };

        let translated = translate_message(msg).unwrap();

        assert_eq!(translated.len(), 1);
        assert_eq!(
            translated[0].reasoning_content.as_deref(),
            Some("I should preserve this")
        );
        assert!(matches!(
            translated[0].content,
            Some(openai::MessageContent::Text(_))
        ));
    }

    #[test]
    fn thinking_only_block_still_becomes_assistant_message() {
        let msg = anthropic::Message {
            role: "assistant".to_string(),
            content: anthropic::MessageContent::Blocks(vec![anthropic::ContentBlock::Thinking {
                thinking: "hidden chain".to_string(),
            }]),
        };

        let translated = translate_message(msg).unwrap();

        assert_eq!(translated.len(), 1);
        assert!(translated[0].content.is_none());
        assert_eq!(
            translated[0].reasoning_content.as_deref(),
            Some("hidden chain")
        );
    }

    #[test]
    fn normalize_schema_adds_empty_required_to_object_schemas() {
        let schema = json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string", "format": "uri" }
            }
        });

        let cleaned = normalize_schema(schema);

        assert_eq!(cleaned["required"], json!([]));
        assert!(cleaned["properties"]["prompt"].get("format").is_none());
    }

    #[test]
    fn normalize_schema_normalizes_non_array_required() {
        let schema = json!({ "type": "object", "required": null });
        let cleaned = normalize_schema(schema);
        assert_eq!(cleaned["required"], json!([]));
    }

    #[test]
    fn normalize_schema_recursively_processes_all_of() {
        let schema = json!({
            "allOf": [
                { "type": "object", "properties": { "a": { "type": "string", "format": "uri" } } },
                { "type": "object", "properties": { "b": { "type": "integer" } } }
            ]
        });

        let cleaned = normalize_schema(schema);

        assert!(cleaned["allOf"][0]["properties"]["a"]
            .get("format")
            .is_none());
        assert_eq!(cleaned["allOf"][0]["required"], json!([]));
        assert_eq!(cleaned["allOf"][1]["required"], json!([]));
    }

    #[test]
    fn normalize_schema_removes_null_values() {
        let schema = json!({
            "type": "object",
            "description": null,
            "properties": { "a": { "type": "string" } }
        });

        let cleaned = normalize_schema(schema);
        assert!(cleaned.get("description").is_none());
    }

    #[test]
    fn remove_term_case_insensitive_with_flexible_whitespace() {
        let result = remove_term("Avoid destructive operations such as RM\t-rF.", "rm -rf");
        assert_eq!(result, "Avoid destructive operations such as .");
    }

    #[test]
    fn remove_term_respects_word_boundaries() {
        let result = remove_term("farm -rf should not match rm -rf", "rm -rf");
        assert_eq!(result, "farm -rf should not match ");
    }

    #[test]
    fn map_stop_reason_translates_all_known_reasons() {
        assert_eq!(map_stop_reason(Some("stop")), Some("end_turn".to_string()));
        assert_eq!(
            map_stop_reason(Some("tool_calls")),
            Some("tool_use".to_string())
        );
        assert_eq!(
            map_stop_reason(Some("length")),
            Some("max_tokens".to_string())
        );
        assert_eq!(
            map_stop_reason(Some("unknown")),
            Some("end_turn".to_string())
        );
        assert_eq!(map_stop_reason(None), None);
    }

    #[test]
    fn batch_tool_detected() {
        let tool = anthropic::Tool {
            name: "x".into(),
            description: None,
            input_schema: json!({}),
            tool_type: Some("BatchTool".into()),
        };
        assert!(is_batch_tool(&tool));
    }

    #[test]
    fn regular_tool_not_batch() {
        let tool = anthropic::Tool {
            name: "x".into(),
            description: None,
            input_schema: json!({}),
            tool_type: None,
        };
        assert!(!is_batch_tool(&tool));
    }
}
