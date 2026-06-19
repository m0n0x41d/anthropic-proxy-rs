mod cli;
mod config;
mod error;
mod metrics;
mod models;
mod proxy;
mod translate;

use axum::{
    routing::{get, post},
    Extension, Router,
};
use clap::Parser;
use cli::{Cli, Command};
use config::Config;
use daemonize::Daemonize;
use reqwest::Client;
use std::sync::Arc;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(command) = cli.command {
        match command {
            Command::Stop { pid_file } => {
                stop_daemon(&pid_file)?;
                return Ok(());
            }
            Command::Status { pid_file } => {
                check_status(&pid_file)?;
                return Ok(());
            }
        }
    }

    if cli.daemon {
        use std::fs::OpenOptions;

        let stdout = OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/anthropic-proxy.log")?;

        let stderr = OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/anthropic-proxy.log")?;

        let daemonize = Daemonize::new()
            .pid_file(&cli.pid_file)
            .working_directory(std::env::current_dir()?)
            .stdout(stdout)
            .stderr(stderr)
            .umask(0o027);

        match daemonize.start() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("✗ Failed to daemonize: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("✓ Starting proxy in foreground mode");
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> anyhow::Result<()> {
    let mut config = Config::from_env_with_path(cli.config)?;

    if cli.debug {
        config.debug = true;
    }
    if cli.verbose {
        config.verbose = true;
    }
    if let Some(port) = cli.port {
        config.port = port;
    }
    if let Some(bind) = cli.bind {
        let trimmed = bind.trim();
        if !trimmed.is_empty() {
            config.bind = trimmed.to_string();
        }
    }
    if !cli.system_prompt_ignore.is_empty() {
        config.system_prompt_ignore_terms.extend(
            cli.system_prompt_ignore
                .into_iter()
                .map(|term| term.trim().to_string())
                .filter(|term| !term.is_empty()),
        );
        Config::dedupe_ignore_terms(&mut config.system_prompt_ignore_terms);
        config.system_prompt_ignore_matchers =
            Config::compile_ignore_terms(&config.system_prompt_ignore_terms);
    }

    let log_level = if config.verbose {
        tracing::Level::TRACE
    } else if config.debug {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("anthropic_proxy={}", log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Anthropic Proxy v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Bind: {}", config.bind);
    tracing::info!("Port: {}", config.port);
    tracing::info!("Upstream URLs: {}", config.upstream_urls.join("; "));
    tracing::info!(
        "Resolved chat completions URLs: {}",
        config.chat_completions_urls().join("; ")
    );
    tracing::info!(
        "Timeouts: connect=10s, read/idle={}s, non-streaming total={}s",
        config.stream_idle_timeout_secs,
        config.request_timeout_secs
    );
    if let Some(ref model) = config.reasoning_model {
        tracing::info!("Reasoning Model Override: {}", model);
    }
    if let Some(ref model) = config.completion_model {
        tracing::info!("Completion Model Override: {}", model);
    }
    if config.passthrough_api_key {
        tracing::info!("API Key: passthrough mode (extracted from x-api-key header)");
    } else if config.api_key.is_some() {
        tracing::info!("API Key: configured");
    } else {
        tracing::info!("API Key: not set (using unauthenticated endpoint)");
    }
    if !config.system_prompt_ignore_matchers.is_empty() {
        let terms = config
            .system_prompt_ignore_matchers
            .iter()
            .map(|term| term.describe())
            .collect::<Vec<_>>()
            .join("; ");
        tracing::info!("System prompt ignore terms: {}", terms);
    }
    if !config.model_map.is_empty() {
        let entries = config
            .model_map
            .iter()
            .map(|(source, target)| format!("{source} -> {target}"))
            .collect::<Vec<_>>()
            .join("; ");
        tracing::info!("Model map: {}", entries);
    }

    let metrics_handle = metrics::install();

    // NB: no total request timeout here. A total timeout caps the *entire*
    // streamed response, so a long-but-active turn (e.g. extended thinking that
    // runs for several minutes) would be aborted mid-stream and surface to the
    // client as "Stream error: error decoding response body". Instead we rely on
    // a read/idle timeout, which resets on every received chunk and only fires
    // when the upstream actually stalls. Non-streaming requests get an explicit
    // total timeout applied per-request in the proxy handler.
    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(std::time::Duration::from_secs(
            config.stream_idle_timeout_secs,
        ))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .pool_max_idle_per_host(10)
        .build()?;

    let config = Arc::new(config);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/messages", post(proxy::proxy_handler))
        .route("/v1/models", get(proxy::list_models_handler))
        .route("/health", axum::routing::get(health_handler))
        .route(
            "/metrics",
            get(move || {
                let handle = metrics_handle.clone();
                async move { handle.render() }
            }),
        )
        .layer(Extension(config.clone()))
        .layer(Extension(client))
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    let addr = format!("{}:{}", config.bind, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    if config.bind == "0.0.0.0" {
        tracing::warn!(
            "Binding to 0.0.0.0 exposes the proxy on every network interface. \
             The proxy may hold an Anthropic API key, which makes this risky on shared networks. \
             Set --bind 127.0.0.1 (or ANTHROPIC_PROXY_BIND=127.0.0.1) to restrict to localhost."
        );
    }

    tracing::info!("Listening on {}", addr);
    tracing::info!("Proxy ready to accept requests");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler() -> &'static str {
    "OK"
}

fn stop_daemon(pid_file: &std::path::Path) -> anyhow::Result<()> {
    if !pid_file.exists() {
        eprintln!("✗ PID file not found: {}", pid_file.display());
        eprintln!("  Daemon is not running or PID file was removed");
        std::process::exit(1);
    }

    let pid_str = std::fs::read_to_string(pid_file)?;
    let pid: i32 = pid_str
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid PID in file: {}", pid_str))?;

    #[cfg(unix)]
    {
        use std::process::Command;
        let output = Command::new("kill").arg(pid.to_string()).output()?;

        if output.status.success() {
            std::fs::remove_file(pid_file)?;
            eprintln!("✓ Daemon stopped (PID: {})", pid);
        } else {
            eprintln!("✗ Failed to stop daemon (PID: {})", pid);
            eprintln!("  Process may have already exited");
            std::fs::remove_file(pid_file)?;
            std::process::exit(1);
        }
    }

    #[cfg(not(unix))]
    {
        eprintln!("✗ Daemon stop is only supported on Unix systems");
        std::process::exit(1);
    }

    Ok(())
}

fn check_status(pid_file: &std::path::Path) -> anyhow::Result<()> {
    if !pid_file.exists() {
        eprintln!("✗ Daemon is not running");
        eprintln!("  PID file not found: {}", pid_file.display());
        std::process::exit(1);
    }

    let pid_str = std::fs::read_to_string(pid_file)?;
    let pid: i32 = pid_str
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid PID in file: {}", pid_str))?;

    #[cfg(unix)]
    {
        use std::process::Command;
        let output = Command::new("ps").arg("-p").arg(pid.to_string()).output()?;

        if output.status.success() {
            eprintln!("✓ Daemon is running (PID: {})", pid);
            eprintln!("  PID file: {}", pid_file.display());
        } else {
            eprintln!("✗ Daemon is not running");
            eprintln!(
                "  Stale PID file found: {} (PID: {})",
                pid_file.display(),
                pid
            );
            std::process::exit(1);
        }
    }

    #[cfg(not(unix))]
    {
        eprintln!("✗ Daemon status check is only supported on Unix systems");
        std::process::exit(1);
    }

    Ok(())
}
