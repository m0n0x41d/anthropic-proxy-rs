use crate::models::anthropic::{
    ContentBlockStart, Delta, DeltaUsage, ErrorData, MessageDeltaData, MessageStartData,
    StreamEvent,
};
use crate::models::openai;
use crate::translate::core;

#[derive(Debug)]
enum BlockState {
    Idle,
    Thinking { index: usize },
    Text { index: usize },
    ToolUse { index: usize },
}

impl BlockState {
    fn current_index(&self) -> Option<usize> {
        match self {
            Self::Idle => None,
            Self::Thinking { index } | Self::Text { index } | Self::ToolUse { index } => {
                Some(*index)
            }
        }
    }
}

#[derive(Debug)]
pub struct StreamState {
    message_id: Option<String>,
    model: Option<String>,
    fallback_model: String,
    block: BlockState,
    next_index: usize,
    message_started: bool,
    /// Set once `translate_done` has emitted the closing events. Guards against a
    /// second call producing a duplicate `message_stop` on a stream that is
    /// already closed.
    finished: bool,
    /// OpenAI delivers usage in a final chunk carrying `choices: []`, which
    /// arrives *after* the chunk with `finish_reason`. Stash it here so
    /// `translate_done` can put the real numbers in `message_delta`.
    pending_usage: Option<openai::Usage>,
    pending_stop_reason: Option<String>,
}

pub fn initial_state(fallback_model: String) -> StreamState {
    StreamState {
        message_id: None,
        model: None,
        fallback_model,
        block: BlockState::Idle,
        next_index: 0,
        message_started: false,
        finished: false,
        pending_usage: None,
        pending_stop_reason: None,
    }
}

/// Build the `message_start` event for whatever the stream has learned so far.
/// Shared by `translate_chunk` (first content chunk) and `translate_done`
/// (a stream that ended before any content chunk ever arrived).
fn message_start_event(state: &StreamState) -> StreamEvent {
    StreamEvent::MessageStart {
        message: MessageStartData {
            id: state
                .message_id
                .clone()
                .unwrap_or_else(|| "msg_proxy".to_string()),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            model: state
                .model
                .clone()
                .unwrap_or_else(|| state.fallback_model.clone()),
            // No usage carried here. See `MessageStartData::usage` -- the honest
            // value at this point is "unknown", and only an absent field says that.
            usage: None,
        },
    }
}

pub fn translate_chunk(state: &mut StreamState, chunk: &openai::StreamChunk) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    if let Some(id) = &chunk.id {
        if state.message_id.is_none() {
            state.message_id = Some(id.clone());
        }
    }
    if let Some(model) = &chunk.model {
        if state.model.is_none() {
            state.model = Some(model.clone());
        }
    }

    // The usage chunk carries no choices, so it must be read *before* the
    // `choices.first()` bail-out below. Reading it after that point -- or only
    // off the finish_reason chunk, where it is null -- discards the one chunk
    // that carries the numbers, and every downstream consumer (cost, context
    // occupancy, auto-compact) is left reading zeros.
    if chunk.usage.is_some() {
        state.pending_usage = chunk.usage.clone();
    }

    let Some(choice) = chunk.choices.first() else {
        return events;
    };

    if !state.message_started {
        events.push(message_start_event(state));
        state.message_started = true;
    }

    for reasoning in [&choice.delta.reasoning, &choice.delta.reasoning_content]
        .into_iter()
        .flatten()
    {
        emit_reasoning(&mut events, state, reasoning);
    }

    if let Some(content) = &choice.delta.content {
        if !content.is_empty() {
            emit_text(&mut events, state, content);
        }
    }

    if let Some(tool_calls) = &choice.delta.tool_calls {
        emit_tool_calls(&mut events, state, tool_calls);
    }

    if let Some(finish_reason) = &choice.finish_reason {
        // Do NOT emit `message_delta` here. The usage chunk has not arrived yet,
        // and `message_delta` is the only place its numbers can go. Record the
        // stop reason and let `translate_done` emit once the usage is in hand.
        close_current_block(&mut events, state);
        state.pending_stop_reason = core::map_stop_reason(Some(finish_reason));
    }

    events
}

pub fn translate_done(state: &mut StreamState) -> Vec<StreamEvent> {
    // Idempotent: a second call has nothing left to close and must not produce a
    // second `message_stop` on an already-finished message.
    if state.finished {
        return Vec::new();
    }
    state.finished = true;

    let mut events = Vec::new();

    // A stream that ended before any content chunk -- e.g. one carrying only a
    // usage chunk, or an empty stream -- never opened the message. Emitting
    // `message_delta` without a preceding `message_start` makes the Anthropic
    // SDK reject the stream outright ("got message_delta before message_start"),
    // so open it here with whatever metadata we have.
    if !state.message_started {
        events.push(message_start_event(state));
        state.message_started = true;
    }

    // A stream that ended without a finish_reason would otherwise leave its
    // content block open; closing here is a no-op when already closed.
    close_current_block(&mut events, state);

    let stop_reason = state
        .pending_stop_reason
        .clone()
        .unwrap_or_else(|| "end_turn".to_string());

    events.push(StreamEvent::MessageDelta {
        delta: MessageDeltaData {
            stop_reason: Some(stop_reason),
            stop_sequence: None,
        },
        usage: DeltaUsage {
            input_tokens: state.pending_usage.as_ref().map(|u| u.prompt_tokens),
            output_tokens: state
                .pending_usage
                .as_ref()
                .map(|u| u.completion_tokens)
                .unwrap_or(0),
        },
    });
    events.push(StreamEvent::MessageStop);
    events
}

pub fn translate_error(message: String) -> Vec<StreamEvent> {
    vec![StreamEvent::Error {
        error: ErrorData {
            error_type: "stream_error".to_string(),
            message,
        },
    }]
}

fn close_current_block(events: &mut Vec<StreamEvent>, state: &mut StreamState) {
    if let Some(index) = state.block.current_index() {
        events.push(StreamEvent::ContentBlockStop { index });
        state.next_index = index + 1;
        // Reset, or a second call re-emits `content_block_stop` for a block
        // that is already closed. `translate_done` relies on this being a no-op
        // on the normal path, where finish_reason closed the block already.
        state.block = BlockState::Idle;
    }
}

fn emit_reasoning(events: &mut Vec<StreamEvent>, state: &mut StreamState, reasoning: &str) {
    if !matches!(state.block, BlockState::Thinking { .. }) {
        close_current_block(events, state);
        let index = state.next_index;
        events.push(StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlockStart::Thinking {
                thinking: String::new(),
            },
        });
        state.block = BlockState::Thinking { index };
    }

    if let BlockState::Thinking { index } = state.block {
        events.push(StreamEvent::ContentBlockDelta {
            index,
            delta: Delta::ThinkingDelta {
                thinking: reasoning.to_string(),
            },
        });
    }
}

fn emit_text(events: &mut Vec<StreamEvent>, state: &mut StreamState, content: &str) {
    if !matches!(state.block, BlockState::Text { .. }) {
        close_current_block(events, state);
        let index = state.next_index;
        events.push(StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlockStart::Text {
                text: String::new(),
            },
        });
        state.block = BlockState::Text { index };
    }

    if let BlockState::Text { index } = state.block {
        events.push(StreamEvent::ContentBlockDelta {
            index,
            delta: Delta::TextDelta {
                text: content.to_string(),
            },
        });
    }
}

fn emit_tool_calls(
    events: &mut Vec<StreamEvent>,
    state: &mut StreamState,
    tool_calls: &[openai::DeltaToolCall],
) {
    for tool_call in tool_calls {
        if let Some(id) = &tool_call.id {
            close_current_block(events, state);
            let index = state.next_index;

            if let Some(function) = &tool_call.function {
                if let Some(name) = &function.name {
                    events.push(StreamEvent::ContentBlockStart {
                        index,
                        content_block: ContentBlockStart::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                        },
                    });
                    state.block = BlockState::ToolUse { index };
                }
            }
        }

        if let Some(function) = &tool_call.function {
            if let Some(args) = &function.arguments {
                if let BlockState::ToolUse { index } = state.block {
                    events.push(StreamEvent::ContentBlockDelta {
                        index,
                        delta: Delta::InputJsonDelta {
                            partial_json: args.clone(),
                        },
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text_chunk(id: &str, model: &str, content: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "content": content } }]
        }))
        .unwrap()
    }

    fn reasoning_chunk(id: &str, model: &str, reasoning: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning": reasoning } }]
        }))
        .unwrap()
    }

    fn reasoning_content_chunk(id: &str, model: &str, reasoning: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning_content": reasoning } }]
        }))
        .unwrap()
    }

    fn finish_chunk(id: &str, model: &str, reason: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }]
        }))
        .unwrap()
    }

    /// The shape OpenAI actually sends with `stream_options.include_usage`: a
    /// trailing chunk with NO choices, carrying only the numbers. The fixture
    /// this replaces put `usage` on the finish_reason chunk -- a shape no
    /// upstream emits -- which is exactly why the suite stayed green while the
    /// numbers were being dropped on the floor.
    fn usage_only_chunk(
        id: &str,
        model: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }))
        .unwrap()
    }

    fn tool_start_chunk(id: &str, model: &str, tool_id: &str, name: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 0, "id": tool_id, "type": "function",
                    "function": { "name": name } }]
            }}]
        }))
        .unwrap()
    }

    fn tool_args_chunk(id: &str, model: &str, args: &str) -> openai::StreamChunk {
        serde_json::from_value(json!({
            "id": id, "model": model,
            "choices": [{ "index": 0, "delta": {
                "tool_calls": [{ "index": 0, "function": { "arguments": args } }]
            }}]
        }))
        .unwrap()
    }

    fn event_types(events: &[StreamEvent]) -> Vec<&str> {
        events.iter().map(|e| e.event_type()).collect()
    }

    #[test]
    fn text_stream_produces_correct_event_sequence() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Hello"));
        assert_eq!(
            event_types(&e1),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );

        let e2 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", " world"));
        assert_eq!(event_types(&e2), ["content_block_delta"]);

        let e3 = translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "stop"));
        assert_eq!(event_types(&e3), ["content_block_stop"]);

        let e4 = translate_done(&mut state);
        assert_eq!(event_types(&e4), ["message_delta", "message_stop"]);
    }

    #[test]
    fn thinking_then_text_produces_two_blocks() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(&mut state, &reasoning_chunk("1", "gpt-4o", "Let me think"));
        assert_eq!(
            event_types(&e1),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );

        let e2 = translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Answer: 42"));
        assert_eq!(
            event_types(&e2),
            [
                "content_block_stop",
                "content_block_start",
                "content_block_delta"
            ]
        );

        if let StreamEvent::ContentBlockStart { index, .. } = &e2[1] {
            assert_eq!(*index, 1);
        }
    }

    #[test]
    fn reasoning_content_produces_thinking_block() {
        let mut state = initial_state("fallback".into());

        let events = translate_chunk(&mut state, &reasoning_content_chunk("1", "gpt-4o", "Think"));

        assert_eq!(
            event_types(&events),
            [
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
        if let StreamEvent::ContentBlockDelta { delta, .. } = &events[2] {
            assert!(matches!(delta, Delta::ThinkingDelta { thinking } if thinking == "Think"));
        }
    }

    #[test]
    fn tool_call_stream() {
        let mut state = initial_state("fallback".into());

        let e1 = translate_chunk(
            &mut state,
            &tool_start_chunk("1", "gpt-4o", "call_abc", "read_file"),
        );
        assert_eq!(event_types(&e1), ["message_start", "content_block_start"]);

        if let StreamEvent::ContentBlockStart { content_block, .. } = &e1[1] {
            match content_block {
                ContentBlockStart::ToolUse { id, name } => {
                    assert_eq!(id, "call_abc");
                    assert_eq!(name, "read_file");
                }
                _ => panic!("expected tool_use block"),
            }
        }

        let e2 = translate_chunk(
            &mut state,
            &tool_args_chunk("1", "gpt-4o", "{\"path\":\"/tmp\"}"),
        );
        assert_eq!(event_types(&e2), ["content_block_delta"]);

        let e3 = translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "tool_calls"));
        assert_eq!(event_types(&e3), ["content_block_stop"]);

        let e4 = translate_done(&mut state);
        assert_eq!(event_types(&e4), ["message_delta", "message_stop"]);

        if let StreamEvent::MessageDelta { delta, .. } = &e4[0] {
            assert_eq!(delta.stop_reason.as_deref(), Some("tool_use"));
        }
    }

    #[test]
    fn usage_arriving_after_finish_reason_reaches_message_delta() {
        let mut state = initial_state("fallback".into());

        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "Hello"));

        // finish_reason closes the block but must NOT emit message_delta yet.
        let finish = translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "stop"));
        assert_eq!(event_types(&finish), ["content_block_stop"]);

        // The usage chunk carries no choices and emits nothing of its own.
        let usage_chunk = translate_chunk(&mut state, &usage_only_chunk("1", "gpt-4o", 7, 3));
        assert!(
            usage_chunk.is_empty(),
            "usage-only chunk must not emit events, but got {:?}",
            event_types(&usage_chunk)
        );

        let events = translate_done(&mut state);
        assert_eq!(event_types(&events), ["message_delta", "message_stop"]);

        if let StreamEvent::MessageDelta { delta, usage } = &events[0] {
            assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(usage.input_tokens, Some(7));
            assert_eq!(usage.output_tokens, 3);
        } else {
            panic!("expected message_delta");
        }
    }

    #[test]
    fn usage_is_kept_whatever_order_it_arrives_in() {
        let mut state = initial_state("fallback".into());

        // Upstream that reports usage early, then finishes.
        translate_chunk(&mut state, &usage_only_chunk("1", "gpt-4o", 11, 5));
        translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "length"));

        let events = translate_done(&mut state);
        if let StreamEvent::MessageDelta { delta, usage } = &events[0] {
            assert_eq!(delta.stop_reason.as_deref(), Some("max_tokens"));
            assert_eq!(usage.input_tokens, Some(11));
            assert_eq!(usage.output_tokens, 5);
        } else {
            panic!("expected message_delta");
        }
    }

    #[test]
    fn done_without_finish_reason_still_emits_message_delta() {
        let mut state = initial_state("fallback".into());
        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "truncated"));

        let events = translate_done(&mut state);
        assert_eq!(
            event_types(&events),
            ["content_block_stop", "message_delta", "message_stop"]
        );
        if let StreamEvent::MessageDelta { delta, usage } = &events[1] {
            assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(usage.input_tokens, None);
            assert_eq!(usage.output_tokens, 0);
        } else {
            panic!("expected message_delta");
        }
    }

    #[test]
    fn done_with_no_prior_chunk_emits_well_formed_message_start() {
        let mut state = initial_state("my-fallback".into());

        let events = translate_done(&mut state);
        assert_eq!(
            event_types(&events),
            ["message_start", "message_delta", "message_stop"]
        );

        if let StreamEvent::MessageStart { message } = &events[0] {
            assert_eq!(message.id, "msg_proxy");
            assert_eq!(message.message_type, "message");
            assert_eq!(message.role, "assistant");
            assert_eq!(message.model, "my-fallback");
            assert!(
                message.usage.is_none(),
                "message_start must not carry a zero-filled usage; the real numbers \
                 belong to message_delta"
            );
        } else {
            panic!("expected message_start");
        }
    }

    #[test]
    fn usage_only_stream_still_opens_with_message_start() {
        // The real-world shape: upstream sends nothing but the trailing usage
        // chunk, which carries no choices and so never opened the message.
        let mut state = initial_state("fallback".into());

        translate_chunk(&mut state, &usage_only_chunk("chatcmpl-9", "gpt-4o", 21, 4));

        let events = translate_done(&mut state);
        assert_eq!(
            event_types(&events),
            ["message_start", "message_delta", "message_stop"]
        );

        if let StreamEvent::MessageStart { message } = &events[0] {
            assert_eq!(message.id, "chatcmpl-9");
            assert_eq!(message.model, "gpt-4o");
        } else {
            panic!("expected message_start");
        }

        if let StreamEvent::MessageDelta { usage, .. } = &events[1] {
            assert_eq!(usage.input_tokens, Some(21));
            assert_eq!(usage.output_tokens, 4);
        } else {
            panic!("expected message_delta");
        }
    }

    #[test]
    fn translate_done_is_idempotent() {
        let mut state = initial_state("fallback".into());
        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "hi"));
        translate_chunk(&mut state, &finish_chunk("1", "gpt-4o", "stop"));

        let first = translate_done(&mut state);
        assert_eq!(event_types(&first), ["message_delta", "message_stop"]);

        let second = translate_done(&mut state);
        assert!(
            second.is_empty(),
            "second translate_done must emit nothing, but got {:?}",
            event_types(&second)
        );
    }

    #[test]
    fn translate_done_is_idempotent_even_when_it_opened_the_message_itself() {
        let mut state = initial_state("fallback".into());

        let first = translate_done(&mut state);
        assert_eq!(
            event_types(&first),
            ["message_start", "message_delta", "message_stop"]
        );

        let second = translate_done(&mut state);
        assert!(
            second.is_empty(),
            "second translate_done must emit nothing, but got {:?}",
            event_types(&second)
        );
    }

    #[test]
    fn text_then_tool_call() {
        let mut state = initial_state("fallback".into());

        translate_chunk(&mut state, &text_chunk("1", "gpt-4o", "I'll read that."));

        let e2 = translate_chunk(
            &mut state,
            &tool_start_chunk("1", "gpt-4o", "call_xyz", "read_file"),
        );

        assert!(event_types(&e2).contains(&"content_block_stop"));
        assert!(event_types(&e2).contains(&"content_block_start"));
    }

    #[test]
    fn message_start_uses_chunk_metadata() {
        let mut state = initial_state("my-fallback".into());

        let events = translate_chunk(&mut state, &text_chunk("chatcmpl-42", "gpt-4o", "hi"));

        if let StreamEvent::MessageStart { message } = &events[0] {
            assert_eq!(message.id, "chatcmpl-42");
            assert_eq!(message.model, "gpt-4o");
            assert_eq!(message.role, "assistant");
        }
    }

    #[test]
    fn fallback_model_used_when_chunk_omits_model() {
        let mut state = initial_state("my-fallback".into());

        let chunk: openai::StreamChunk = serde_json::from_value(json!({
            "choices": [{ "index": 0, "delta": { "content": "hey" } }]
        }))
        .unwrap();

        let events = translate_chunk(&mut state, &chunk);

        if let StreamEvent::MessageStart { message } = &events[0] {
            assert_eq!(message.model, "my-fallback");
        }
    }

    #[test]
    fn error_event_produced() {
        let events = translate_error("connection reset".into());
        assert_eq!(event_types(&events), ["error"]);

        if let StreamEvent::Error { error } = &events[0] {
            assert!(error.message.contains("connection reset"));
        }
    }

    #[test]
    fn empty_content_not_emitted() {
        let mut state = initial_state("fallback".into());

        let chunk: openai::StreamChunk = serde_json::from_value(json!({
            "id": "1", "model": "gpt-4o",
            "choices": [{ "index": 0, "delta": { "content": "" } }]
        }))
        .unwrap();

        let events = translate_chunk(&mut state, &chunk);

        let deltas: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ContentBlockDelta { .. }))
            .collect();
        assert!(deltas.is_empty());
    }
}
