use clap::Parser;
use std::path::PathBuf;

/// High-performance Anthropic API proxy to OpenAI-compatible endpoints
#[derive(Parser, Debug)]
#[command(
    name = "anthropic-proxy",
    version,
    about = "Proxy Anthropic API requests to OpenAI-compatible endpoints",
    long_about = "A high-performance proxy that translates Anthropic Claude API requests \
                  to OpenAI-compatible endpoints like OpenRouter, allowing you to use \
                  Claude-compatible clients with any OpenAI-compatible API."
)]
pub struct Cli {
    /// Path to custom .env configuration file
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Enable debug logging (same as DEBUG=true)
    #[arg(short, long)]
    pub debug: bool,

    /// Enable verbose logging (logs full request/response bodies)
    #[arg(short, long)]
    pub verbose: bool,

    /// Port to listen on (overrides PORT env var)
    #[arg(short, long, value_name = "PORT")]
    pub port: Option<u16>,
}
