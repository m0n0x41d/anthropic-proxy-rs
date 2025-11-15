use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    #[command(subcommand)]
    pub command: Option<Command>,

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

    /// Run as background daemon
    #[arg(long)]
    pub daemon: bool,

    /// PID file path (used with daemon commands)
    #[arg(long, value_name = "FILE", default_value = "/tmp/anthropic-proxy.pid")]
    pub pid_file: PathBuf,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Stop running daemon
    Stop {
        /// PID file path
        #[arg(long, value_name = "FILE", default_value = "/tmp/anthropic-proxy.pid")]
        pid_file: PathBuf,
    },
    /// Check daemon status
    Status {
        /// PID file path
        #[arg(long, value_name = "FILE", default_value = "/tmp/anthropic-proxy.pid")]
        pid_file: PathBuf,
    },
}
