use anyhow::Result;
use std::{env, path::PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub base_url: String,
    pub api_key: Option<String>,
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub debug: bool,
    pub verbose: bool,
}

impl Config {
    fn load_dotenv(custom_path: Option<PathBuf>) -> Option<PathBuf> {
        if let Some(path) = custom_path {
            if path.exists() {
                if let Ok(_) = dotenvy::from_path(&path) {
                    return Some(path);
                }
            }
            eprintln!("⚠️  WARNING: Custom config file not found: {}", path.display());
        }

        if let Ok(path) = dotenvy::dotenv() {
            return Some(path);
        }

        if let Some(home) = env::var("HOME").ok() {
            let home_config = PathBuf::from(home).join(".anthropic-proxy.env");
            if home_config.exists() {
                if let Ok(_) = dotenvy::from_path(&home_config) {
                    return Some(home_config);
                }
            }
        }

        let etc_config = PathBuf::from("/etc/anthropic-proxy/.env");
        if etc_config.exists() {
            if let Ok(_) = dotenvy::from_path(&etc_config) {
                return Some(etc_config);
            }
        }

        None
    }

    pub fn from_env() -> Result<Self> {
        Self::from_env_with_path(None)
    }

    pub fn from_env_with_path(custom_path: Option<PathBuf>) -> Result<Self> {
        if let Some(path) = Self::load_dotenv(custom_path) {
            eprintln!("📄 Loaded config from: {}", path.display());
        } else {
            eprintln!("ℹ️  No .env file found, using environment variables only");
        }

        let port = env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);

        let base_url = env::var("UPSTREAM_BASE_URL")
            .or_else(|_| env::var("ANTHROPIC_PROXY_BASE_URL"))
            .map_err(|_| anyhow::anyhow!(
                "UPSTREAM_BASE_URL is required. Set it to your OpenAI-compatible endpoint.\n\
                Examples:\n\
                  - OpenRouter: https://openrouter.ai/api\n\
                  - OpenAI: https://api.openai.com\n\
                  - Local: http://localhost:11434"
            ))?;

        let api_key = env::var("UPSTREAM_API_KEY")
            .or_else(|_| env::var("OPENROUTER_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty());

        let reasoning_model = env::var("REASONING_MODEL").ok();
        let completion_model = env::var("COMPLETION_MODEL").ok();

        let debug = env::var("DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let verbose = env::var("VERBOSE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        if base_url.ends_with("/v1") {
            eprintln!("⚠️  WARNING: UPSTREAM_BASE_URL ends with '/v1'");
            eprintln!("   This will result in URLs like: {}/v1/chat/completions", base_url);
            eprintln!("   Consider removing '/v1' from UPSTREAM_BASE_URL");
            eprintln!("   Correct: https://openrouter.ai/api");
            eprintln!("   Wrong:   https://openrouter.ai/api/v1");
        }

        Ok(Config {
            port,
            base_url,
            api_key,
            reasoning_model,
            completion_model,
            debug,
            verbose,
        })
    }

    pub fn chat_completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'))
    }
}
