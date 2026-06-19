# anthropic-proxy-rs

High-performance Rust proxy that translates Anthropic API requests to OpenAI-compatible format. Use Claude Code, Claude Desktop, or any Anthropic API client with OpenRouter, native OpenAI, or any OpenAI-compatible endpoint.

## Features

- **Fast & Lightweight**: Written in Rust with async I/O (~3MB binary)
- **Full Streaming**: Server-Sent Events (SSE) with real-time responses
- **Tool Calling**: Complete support for function/tool calling
- **Universal**: Works with any OpenAI-compatible API (OpenRouter, OpenAI, Azure, local LLMs)
- **Extended Thinking**: Supports Claude's reasoning mode
- **Drop-in Replacement**: Compatible with official Anthropic SDKs

## Quick Start

> **Note**: Using [Task](https://taskfile.dev) is currently recommended. Install with `brew install go-task` (macOS) or see the [installation guide](https://taskfile.dev/installation/). Releases with build binaries will be made soon.


```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build and install to PATH with task

```bash
task local-install
```

### Install with curl

```bash
curl -fsSL https://raw.githubusercontent.com/m0n0x41d/anthropic-proxy-rs/main/install.sh | bash
```

This installer uses `cargo install --git ... --locked` under the hood, so Rust/Cargo still needs to be present on the machine.

# Run from anywhere
```bash
UPSTREAM_BASE_URL=https://openrouter.ai/api \
UPSTREAM_API_KEY=sk-or-... \
anthropic-proxy
```

# Or build and run manually
```bash
cargo build --release
UPSTREAM_BASE_URL=https://api.openai.com \
UPSTREAM_API_KEY=sk-... \
./target/release/anthropic-proxy
```

## Configuration

### Command Line Options

```bash
anthropic-proxy --help
```

**Commands:**
| Command | Description |
|---------|-------------|
| `stop` | Stop running daemon |
| `status` | Check daemon status |

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--config <FILE>` | `-c` | Path to custom .env file |
| `--debug` | `-d` | Enable debug logging |
| `--verbose` | `-v` | Enable verbose logging (logs full request/response bodies) |
| `--port <PORT>` | `-p` | Port to listen on (overrides PORT env var) |
| `--bind <ADDR>` | | Address to bind the listener to (overrides `ANTHROPIC_PROXY_BIND`, default `0.0.0.0`) |
| `--system-prompt-ignore <TEXT>` | | Remove one or more system prompt terms before forwarding upstream (repeat or separate with `;`; wrap in `/.../` for a regex) |
| `--daemon` | | Run as background daemon |
| `--pid-file <FILE>` | | PID file path (default: `/tmp/anthropic-proxy.pid`) |
| `--help` | `-h` | Print help information |
| `--version` | `-V` | Print version |

### Environment Variables

Configuration can be set via environment variables or `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `UPSTREAM_BASE_URL` | **Yes** | - | OpenAI-compatible endpoint URL |
| `UPSTREAM_API_KEY` | No* | - | API key for upstream service |
| `UPSTREAM_API_KEY_PASSTHROUGH` | No | `false` | Extract API key from incoming `x-api-key` header per request (`true`/`false`) |
| `PORT` | No | `3000` | Server port |
| `ANTHROPIC_PROXY_BIND` | No | `0.0.0.0` | Listener bind address. Set to `127.0.0.1` to restrict access to localhost (recommended on shared networks). When bound to `0.0.0.0`, a warning is logged. |
| `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS` | No | - | System prompt terms to remove before forwarding upstream (`;` or newline separated; wrap in `/.../` for a regex) |
| `ANTHROPIC_PROXY_MODEL_MAP` | No | - | Exact model remapping before the upstream call (`source=target;other=target`) |
| `REASONING_MODEL` | No | (uses request model) | Model to use when extended thinking is enabled** |
| `COMPLETION_MODEL` | No | (uses request model) | Model to use for standard requests (no thinking)** |
| `ANTHROPIC_PROXY_REQUEST_TIMEOUT` | No | `300` | Total timeout (seconds) for **non-streaming** upstream requests. Streaming requests have no total timeout (see `ANTHROPIC_PROXY_STREAM_IDLE_TIMEOUT`). |
| `ANTHROPIC_PROXY_STREAM_IDLE_TIMEOUT` | No | `300` | Read/idle timeout (seconds) for upstream connections. Resets on every received chunk, so it only fires when the upstream goes silent — it does **not** cap the total duration of a long, actively-streaming response. Raise it if your upstream has long quiet gaps during extended thinking. |
| `DEBUG` | No | `false` | Enable debug logging (`1` or `true`) |
| `VERBOSE` | No | `false` | Enable verbose logging (`1` or `true`) |

\* Required if your upstream endpoint needs authentication  
\*\* The proxy automatically detects when a request has extended thinking enabled (via the `thinking` parameter in the request) and routes it to `REASONING_MODEL`. Standard requests without thinking use `COMPLETION_MODEL`. This allows you to use more powerful models for reasoning tasks and faster/cheaper models for simple completions. If not set, the model from the client request is used.

`UPSTREAM_BASE_URL` accepts any of these forms:
- Service base URL: `https://api.openai.com` -> `/v1/chat/completions`
- Versioned base URL: `https://gateway.company.internal/v2` -> `/v2/chat/completions`
- Full endpoint: `https://gateway.company.internal/v2/chat/completions`

System prompt sanitization:
- The proxy can remove configured terms from upstream `system` prompts before forwarding.
- Set terms with `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS='rm -rf;git reset --hard'`
- Or repeat `--system-prompt-ignore`, for example `--system-prompt-ignore 'rm -rf' --system-prompt-ignore 'git reset --hard'`
- Plain terms match literally (case-insensitive, whitespace-flexible, on word boundaries).
- A term wrapped in slashes is treated as a regular expression, for example `--system-prompt-ignore '/sk-[A-Za-z0-9]{20,}/'` to strip API-key-like strings. Add the inline `(?i)` flag for case-insensitive matching, e.g. `/(?i)do not refuse/`. Every match is removed.
- Regex patterns cannot contain the `;` separator literally (it splits terms); pass such a pattern via a dedicated `--system-prompt-ignore` flag or escape it as `\x3b`. Invalid patterns are reported on startup and skipped.

Model mapping:
- `ANTHROPIC_PROXY_MODEL_MAP='claude-opus-4-6=openai/gpt-4.1;claude-haiku-4-5=openai/gpt-4.1-mini'`
- `REASONING_MODEL` and `COMPLETION_MODEL` are selected first, then `ANTHROPIC_PROXY_MODEL_MAP` is applied to the final model name before the upstream call

### Configuration File Locations

The proxy searches for `.env` files in the following order:

1. Custom path specified with `--config` flag
2. Current working directory (`./.env`)
3. User home directory (`~/.anthropic-proxy.env`)
4. System-wide config (`/etc/anthropic-proxy/.env`)

If no `.env` file is found, the proxy uses environment variables from your shell.

### API Key Passthrough

When `UPSTREAM_API_KEY_PASSTHROUGH=true` is set, the proxy extracts the API key from each incoming request's `x-api-key` header (the standard header used by Anthropic SDKs and clients) and forwards it as `Authorization: Bearer {key}` to the upstream OpenAI-compatible endpoint.

This is useful when you want each client to authenticate with its own key to the upstream service, rather than using a single static key configured in `UPSTREAM_API_KEY`.

```bash
# Enable passthrough mode (UPSTREAM_API_KEY must NOT be set)
UPSTREAM_API_KEY_PASSTHROUGH=true \
UPSTREAM_BASE_URL=https://openrouter.ai/api \
anthropic-proxy
```

**Important constraints:**
- `UPSTREAM_API_KEY_PASSTHROUGH=true` **cannot** be combined with `UPSTREAM_API_KEY`. If both are set, the proxy will refuse to start.
- If passthrough is enabled but the incoming request has no `x-api-key` header (or an empty one), no `Authorization` header is sent upstream — the upstream endpoint decides whether to accept unauthenticated requests.
- Passthrough applies to both `/v1/messages` and `/v1/models` endpoints, as both receive the `x-api-key` header from Anthropic clients.

## Usage Examples

### With Claude Code

```bash
# Start proxy as daemon and use Claude Code immediately
anthropic-proxy --daemon && ANTHROPIC_BASE_URL=http://localhost:3000 claude

# Or use separate terminals:
# Terminal 1: Start proxy
anthropic-proxy

# Terminal 2: Use Claude Code
ANTHROPIC_BASE_URL=http://localhost:3000 claude
```

### With Debug Logging

```bash
# Enable debug logging via CLI flag
anthropic-proxy --debug

# Or via environment variable
DEBUG=true anthropic-proxy

# Enable verbose logging (logs full request/response bodies)
anthropic-proxy --verbose
```

### With System Prompt Ignore Terms

```bash
# Remove specific terms via environment variable
ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS='rm -rf;git reset --hard' anthropic-proxy

# Or via CLI flag
anthropic-proxy \
  --system-prompt-ignore 'rm -rf' \
  --system-prompt-ignore 'git reset --hard'
```

### With Custom Config File

```bash
# Use a custom .env file
anthropic-proxy --config /path/to/my-config.env

# Or place it in your home directory
cp .env ~/.anthropic-proxy.env
anthropic-proxy
```

### With Custom Model Overrides

```bash
# Use different models for reasoning vs standard completion
# Reasoning model is used when extended thinking is enabled in the request
# Completion model is used for standard requests without thinking
UPSTREAM_BASE_URL=https://openrouter.ai/api \
  UPSTREAM_API_KEY=sk-or-... \
  REASONING_MODEL=anthropic/claude-3.5-sonnet \
  COMPLETION_MODEL=anthropic/claude-3-haiku \
  PORT=8080 \
  anthropic-proxy

# This allows cost optimization: use powerful models for complex reasoning,
# and faster/cheaper models for simple completions
```

### Running as Daemon

```bash
# Start as background daemon
anthropic-proxy --daemon

# Check daemon status
anthropic-proxy status

# Stop daemon
anthropic-proxy stop

# View daemon logs
tail -f /tmp/anthropic-proxy.log

# Custom PID file location
anthropic-proxy --daemon --pid-file ~/.anthropic-proxy.pid
anthropic-proxy stop --pid-file ~/.anthropic-proxy.pid
```

> **Note**: When running as daemon, logs are written to `/tmp/anthropic-proxy.log`

### With Model Mapping

```bash
UPSTREAM_BASE_URL=https://gateway.company.internal/v2 \
  UPSTREAM_API_KEY=sk-... \
  ANTHROPIC_PROXY_MODEL_MAP='claude-opus-4-6=openai/gpt-4.1;claude-haiku-4-5=openai/gpt-4.1-mini' \
  anthropic-proxy
```

## Supported Features

✅ Text messages  
✅ System prompts (single and multiple)  
✅ Image content (base64)  
✅ Tool/function calling  
✅ Tool results  
✅ Streaming responses  
✅ Extended thinking mode (automatic model routing)  
✅ Temperature, top_p, top_k  
✅ Stop sequences  
✅ Max tokens  

> **Note**: Make sure your upstream model supports tool use. Especially if you are using this proxy for coding agents like Claude Code.

### Extended Thinking Mode

The proxy automatically detects when a request includes the `thinking` parameter (Claude Codes's for example) and routes it to the model specified in `REASONING_MODEL`. Requests without thinking use `COMPLETION_MODEL`. 

If model override variables are not set, the proxy uses the model specified in the client request.

## Known Limitations
The following Anthropic API features are **not supported** currently (Claude Code and similar tools working without these parameters):):

- `tool_choice` parameter (always uses `auto`)
- `service_tier` parameter
- `metadata` parameter
- `context_management` parameter
- `container` parameter
- Citations in responses
- `pause_turn` and `refusal` stop reasons
- Message Batches API
- Files API
- Admin API

## Troubleshooting & Known Pitfalls

**Error: `UPSTREAM_BASE_URL is required`**  
→ You must set the upstream endpoint URL. Examples:
  - OpenRouter: `https://openrouter.ai/api`
  - OpenAI: `https://api.openai.com`
  - Local: `http://localhost:11434`

**Error: `405 Method Not Allowed` or wrong upstream path**  
→ Check how `UPSTREAM_BASE_URL` is being resolved:
  - `https://api.openai.com` -> `https://api.openai.com/v1/chat/completions`
  - `https://openrouter.ai/api` -> `https://openrouter.ai/api/v1/chat/completions`
  - `https://gateway.company.internal/v2` -> `https://gateway.company.internal/v2/chat/completions`
  - `https://gateway.company.internal/v2/chat/completions` -> used as-is
  - Partial paths like `.../chat` and URLs with query strings/fragments are rejected

**Model not found errors**  
→ Set `REASONING_MODEL` and `COMPLETION_MODEL` to override the models from client requests, or use `ANTHROPIC_PROXY_MODEL_MAP` to remap client model names to upstream model names

**Gateway/WAF blocks Claude Code system prompts with `403`**
→ Use `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS` or `--system-prompt-ignore` to remove offending terms before forwarding upstream

**Error: `Stream error: error decoding response body` after long thinking/generation**
→ The upstream connection stalled or was capped mid-stream. Streaming requests have no total timeout, but a read/idle timeout still applies. If your upstream goes quiet for long stretches during extended thinking, raise `ANTHROPIC_PROXY_STREAM_IDLE_TIMEOUT` (seconds).

## License

MIT License - Copyright (c) 2025 m0n0x41d (Ivan Zakutnii)

See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `cargo test && cargo clippy`
5. Submit a pull request

## Links

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Rust Documentation](https://doc.rust-lang.org/)
