use anyhow::{bail, Result};
use regex::Regex;
use reqwest::Url;
use std::{collections::BTreeMap, env, path::PathBuf};

/// A single system prompt ignore rule, compiled from its raw configuration string.
///
/// A raw term wrapped in slashes (`/pattern/`) is treated as a regular expression;
/// anything else is matched literally (case-insensitive, whitespace-flexible).
#[derive(Debug, Clone)]
pub enum IgnoreTerm {
    Literal(String),
    Regex(Regex),
}

impl IgnoreTerm {
    /// Human-readable form used for logging which terms were applied.
    pub fn describe(&self) -> String {
        match self {
            IgnoreTerm::Literal(term) => term.clone(),
            IgnoreTerm::Regex(re) => format!("/{}/", re.as_str()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub bind: String,
    pub upstream_urls: Vec<String>,
    pub api_key: Option<String>,
    pub passthrough_api_key: bool,
    pub model_map: BTreeMap<String, String>,
    pub system_prompt_ignore_terms: Vec<String>,
    pub system_prompt_ignore_matchers: Vec<IgnoreTerm>,
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub debug: bool,
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 3000,
            bind: "0.0.0.0".to_string(),
            upstream_urls: vec!["http://localhost:11434".to_string()],
            api_key: None,
            passthrough_api_key: false,
            model_map: BTreeMap::new(),
            system_prompt_ignore_terms: Vec::new(),
            system_prompt_ignore_matchers: Vec::new(),
            reasoning_model: None,
            completion_model: None,
            debug: false,
            verbose: false,
        }
    }
}

impl Config {
    fn load_dotenv(custom_path: Option<PathBuf>) -> Option<PathBuf> {
        if let Some(path) = custom_path {
            if path.exists() && dotenvy::from_path(&path).is_ok() {
                return Some(path);
            }
            eprintln!(
                "⚠️  WARNING: Custom config file not found: {}",
                path.display()
            );
        }

        if let Ok(path) = dotenvy::dotenv() {
            return Some(path);
        }

        if let Ok(home) = env::var("HOME") {
            let home_config = PathBuf::from(home).join(".anthropic-proxy.env");
            if home_config.exists() && dotenvy::from_path(&home_config).is_ok() {
                return Some(home_config);
            }
        }

        let etc_config = PathBuf::from("/etc/anthropic-proxy/.env");
        if etc_config.exists() && dotenvy::from_path(&etc_config).is_ok() {
            return Some(etc_config);
        }

        None
    }

    #[allow(dead_code)]
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

        let bind = env::var("ANTHROPIC_PROXY_BIND")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "0.0.0.0".to_string());

        let raw_urls = env::var("UPSTREAM_BASE_URL")
            .or_else(|_| env::var("ANTHROPIC_PROXY_BASE_URL"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "UPSTREAM_BASE_URL is required. Set it to your OpenAI-compatible endpoint.\n\
                Examples:\n\
                  - OpenRouter: https://openrouter.ai/api\n\
                  - OpenAI: https://api.openai.com\n\
                  - Multiple (failover): https://openrouter.ai/api;https://api.openai.com\n\
                  - Local: http://localhost:11434"
                )
            })?;

        let upstream_urls = Self::parse_upstream_urls(&raw_urls)?;

        let api_key = env::var("UPSTREAM_API_KEY")
            .or_else(|_| env::var("OPENROUTER_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty());

        let model_map = env::var("ANTHROPIC_PROXY_MODEL_MAP")
            .ok()
            .map(|value| Self::parse_model_map(&value))
            .transpose()?
            .unwrap_or_default();

        let mut system_prompt_ignore_terms = env::var("ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS")
            .ok()
            .map(|value| Self::parse_system_prompt_ignore_terms(&value))
            .unwrap_or_default();
        Self::dedupe_ignore_terms(&mut system_prompt_ignore_terms);
        let system_prompt_ignore_matchers = Self::compile_ignore_terms(&system_prompt_ignore_terms);

        let reasoning_model = env::var("REASONING_MODEL").ok();
        let completion_model = env::var("COMPLETION_MODEL").ok();

        let debug = env::var("DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let verbose = env::var("VERBOSE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let passthrough_api_key = env::var("UPSTREAM_API_KEY_PASSTHROUGH")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        // Validate: UPSTREAM_API_KEY_PASSTHROUGH requires UPSTREAM_API_KEY to be unset
        if passthrough_api_key && api_key.is_some() {
            bail!(
                "UPSTREAM_API_KEY_PASSTHROUGH=true cannot be used together with UPSTREAM_API_KEY.\n\
                 When passthrough is enabled, the API key is extracted from each incoming request's x-api-key header.\n\
                 Unset UPSTREAM_API_KEY or set UPSTREAM_API_KEY_PASSTHROUGH=false."
            );
        }

        Ok(Config {
            port,
            bind,
            upstream_urls,
            api_key,
            passthrough_api_key,
            model_map,
            system_prompt_ignore_terms,
            system_prompt_ignore_matchers,
            reasoning_model,
            completion_model,
            debug,
            verbose,
        })
    }

    pub fn chat_completions_urls(&self) -> Vec<String> {
        self.upstream_urls
            .iter()
            .map(|url| {
                Self::resolve_chat_completions_url(url)
                    .expect("URLs should be validated during configuration loading")
            })
            .collect()
    }

    pub fn models_urls(&self) -> Vec<String> {
        self.upstream_urls
            .iter()
            .map(|url| {
                Self::resolve_models_url(url)
                    .expect("URLs should be validated during configuration loading")
            })
            .collect()
    }

    fn parse_upstream_urls(raw: &str) -> Result<Vec<String>> {
        let urls: Vec<String> = raw
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();

        if urls.is_empty() {
            bail!("UPSTREAM_BASE_URL must not be empty");
        }

        for url in &urls {
            Self::resolve_chat_completions_url(url)?;
        }

        Ok(urls)
    }

    fn resolve_chat_completions_url(base_url: &str) -> Result<String> {
        let (normalized, path_segments) = Self::parse_base_url(base_url)?;

        if Self::is_chat_completions_path(&path_segments) {
            return Ok(normalized.to_string());
        }

        let last_segment = path_segments.last().map(String::as_str);
        if matches!(last_segment, Some("chat") | Some("completions")) {
            bail!(
                "UPSTREAM_BASE_URL must be either a service base URL, a versioned base URL like https://gateway.example.com/v2, or the full .../chat/completions endpoint"
            );
        }

        if last_segment.is_some_and(Self::is_version_segment) {
            return Ok(format!("{}/chat/completions", normalized));
        }

        Ok(format!("{}/v1/chat/completions", normalized))
    }

    fn resolve_models_url(base_url: &str) -> Result<String> {
        let (normalized, path_segments) = Self::parse_base_url(base_url)?;

        if Self::is_chat_completions_path(&path_segments) {
            let base = normalized
                .trim_end_matches("/chat/completions")
                .trim_end_matches('/');
            return Ok(format!("{}/models", base));
        }

        let last_segment = path_segments.last().map(String::as_str);
        if matches!(last_segment, Some("chat") | Some("completions")) {
            bail!(
                "UPSTREAM_BASE_URL must be either a service base URL, a versioned base URL like https://gateway.example.com/v2, or the full .../chat/completions endpoint"
            );
        }

        if last_segment.is_some_and(Self::is_version_segment) {
            return Ok(format!("{}/models", normalized));
        }

        Ok(format!("{}/v1/models", normalized))
    }

    fn parse_base_url(base_url: &str) -> Result<(String, Vec<String>)> {
        let normalized = base_url.trim();

        if normalized.is_empty() {
            bail!("UPSTREAM_BASE_URL must not be empty");
        }

        let parsed = Url::parse(normalized).map_err(|err| {
            anyhow::anyhow!("UPSTREAM_BASE_URL must be a valid http(s) URL: {}", err)
        })?;

        if !matches!(parsed.scheme(), "http" | "https") {
            bail!("UPSTREAM_BASE_URL must use http or https");
        }

        if parsed.query().is_some() || parsed.fragment().is_some() {
            bail!("UPSTREAM_BASE_URL must not include query parameters or fragments");
        }

        let path_segments: Vec<_> = parsed
            .path_segments()
            .map(|segments| {
                segments
                    .filter(|segment| !segment.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();

        Ok((normalized.trim_end_matches('/').to_string(), path_segments))
    }

    fn is_chat_completions_path(segments: &[String]) -> bool {
        matches!(segments, [.., chat, completions] if chat == "chat" && completions == "completions")
    }

    fn is_version_segment(segment: &str) -> bool {
        let version = segment
            .strip_prefix('v')
            .or_else(|| segment.strip_prefix('V'));

        version
            .is_some_and(|value| !value.is_empty() && value.chars().all(|ch| ch.is_ascii_digit()))
    }

    pub fn parse_system_prompt_ignore_terms(value: &str) -> Vec<String> {
        value
            .split([';', '\n'])
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    }

    /// Compile raw ignore terms into matchers.
    ///
    /// A term wrapped in slashes (`/pattern/`) is compiled as a regular expression.
    /// Use the inline `(?i)` flag for case-insensitive matching. Any other term is
    /// matched literally. Invalid regexes are reported and skipped so a single bad
    /// pattern does not take down the proxy.
    pub fn compile_ignore_terms(terms: &[String]) -> Vec<IgnoreTerm> {
        terms
            .iter()
            .filter_map(|term| Self::compile_ignore_term(term))
            .collect()
    }

    fn compile_ignore_term(term: &str) -> Option<IgnoreTerm> {
        if let Some(pattern) = term
            .strip_prefix('/')
            .and_then(|rest| rest.strip_suffix('/'))
            .filter(|pattern| !pattern.is_empty())
        {
            return match Regex::new(pattern) {
                Ok(regex) => Some(IgnoreTerm::Regex(regex)),
                Err(err) => {
                    eprintln!(
                        "⚠️  WARNING: Ignoring invalid system prompt regex '{}': {}",
                        term, err
                    );
                    None
                }
            };
        }

        Some(IgnoreTerm::Literal(term.to_string()))
    }

    pub fn dedupe_ignore_terms(terms: &mut Vec<String>) {
        let mut deduped = Vec::new();
        let mut seen = Vec::new();
        for term in terms.drain(..) {
            let normalized = term
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_ascii_lowercase();
            if !seen.iter().any(|existing: &String| existing == &normalized) {
                seen.push(normalized);
                deduped.push(term);
            }
        }
        *terms = deduped;
    }

    pub fn parse_model_map(value: &str) -> Result<BTreeMap<String, String>> {
        let mut model_map = BTreeMap::new();

        for entry in value
            .split([';', '\n'])
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
        {
            let (source, target) = entry.split_once('=').ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid ANTHROPIC_PROXY_MODEL_MAP entry '{}'. Expected source=target",
                    entry
                )
            })?;

            let source = source.trim();
            let target = target.trim();

            if source.is_empty() || target.is_empty() {
                bail!(
                    "Invalid ANTHROPIC_PROXY_MODEL_MAP entry '{}'. Source and target models must be non-empty",
                    entry
                );
            }

            model_map.insert(source.to_string(), target.to_string());
        }

        Ok(model_map)
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn base_url_without_version_defaults_to_v1_endpoint() {
        let url = Config::resolve_chat_completions_url("https://api.openai.com").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn versioned_base_url_preserves_existing_version() {
        let url = Config::resolve_chat_completions_url("https://gateway.example.com/v2").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/chat/completions");
    }

    #[test]
    fn full_chat_completions_endpoint_is_used_as_is() {
        let url = Config::resolve_chat_completions_url(
            "https://gateway.example.com/v2/chat/completions/",
        )
        .unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/chat/completions");
    }

    #[test]
    fn models_url_without_version_defaults_to_v1_endpoint() {
        let url = Config::resolve_models_url("https://api.openai.com").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/models");
    }

    #[test]
    fn versioned_models_url_preserves_existing_version() {
        let url = Config::resolve_models_url("https://gateway.example.com/v2").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/models");
    }

    #[test]
    fn full_chat_completions_endpoint_resolves_models_url() {
        let url =
            Config::resolve_models_url("https://gateway.example.com/v2/chat/completions").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/models");
    }

    #[test]
    fn partial_chat_path_is_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2/chat")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("service base URL, a versioned base URL"));
    }

    #[test]
    fn query_strings_are_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2?foo=bar")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn fragments_are_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2#section")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn empty_url_is_rejected() {
        let err = Config::resolve_chat_completions_url("").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn non_http_scheme_is_rejected() {
        let err = Config::resolve_chat_completions_url("ftp://gateway.example.com").unwrap_err();
        assert!(err.to_string().contains("must use http or https"));
    }

    #[test]
    fn explicit_v1_is_preserved_not_doubled() {
        let url = Config::resolve_chat_completions_url("https://openrouter.ai/api/v1").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn trailing_slash_on_base_url_is_normalized() {
        let url = Config::resolve_chat_completions_url("https://api.openai.com/").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn models_url_from_explicit_v1() {
        let url = Config::resolve_models_url("https://openrouter.ai/api/v1").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/models");
    }

    #[test]
    fn models_url_with_trailing_slash() {
        let url = Config::resolve_models_url("https://api.openai.com/").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/models");
    }

    #[test]
    fn url_with_subpath_and_no_version_defaults_to_v1() {
        let url = Config::resolve_chat_completions_url("https://openrouter.ai/api").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn only_completions_path_is_rejected() {
        let err =
            Config::resolve_chat_completions_url("https://gateway.example.com/v2/completions")
                .unwrap_err();
        assert!(err
            .to_string()
            .contains("service base URL, a versioned base URL"));
    }

    #[test]
    fn uppercase_version_prefix_is_accepted() {
        let url = Config::resolve_chat_completions_url("https://gateway.example.com/V2").unwrap();
        assert_eq!(url, "https://gateway.example.com/V2/chat/completions");
    }

    #[test]
    fn parse_system_prompt_ignore_terms_supports_semicolons_and_newlines() {
        let terms =
            Config::parse_system_prompt_ignore_terms("rm -rf;git reset --hard\nsudo rm -rf");

        assert_eq!(
            terms,
            vec![
                "rm -rf".to_string(),
                "git reset --hard".to_string(),
                "sudo rm -rf".to_string()
            ]
        );
    }

    #[test]
    fn dedupe_ignore_terms_normalizes_case_and_whitespace() {
        let mut terms = vec![
            "rm -rf".to_string(),
            " RM\t-rF ".to_string(),
            "git reset --hard".to_string(),
        ];

        Config::dedupe_ignore_terms(&mut terms);

        assert_eq!(
            terms,
            vec!["rm -rf".to_string(), "git reset --hard".to_string()]
        );
    }

    #[test]
    fn compile_ignore_terms_distinguishes_literals_and_regexes() {
        let matchers =
            Config::compile_ignore_terms(&["rm -rf".to_string(), r"/[A-Z]{3}-\d{5}/".to_string()]);

        assert_eq!(matchers.len(), 2);
        assert!(matches!(&matchers[0], super::IgnoreTerm::Literal(term) if term == "rm -rf"));
        assert!(matches!(&matchers[1], super::IgnoreTerm::Regex(_)));
    }

    #[test]
    fn compile_ignore_terms_skips_invalid_regex() {
        let matchers =
            Config::compile_ignore_terms(&["keep".to_string(), "/(unclosed/".to_string()]);

        assert_eq!(matchers.len(), 1);
        assert!(matches!(&matchers[0], super::IgnoreTerm::Literal(term) if term == "keep"));
    }

    #[test]
    fn compile_ignore_terms_treats_empty_slashes_as_literal() {
        let matchers = Config::compile_ignore_terms(&["//".to_string()]);

        assert_eq!(matchers.len(), 1);
        assert!(matches!(&matchers[0], super::IgnoreTerm::Literal(term) if term == "//"));
    }

    #[test]
    fn parse_model_map_supports_semicolons_and_newlines() {
        let model_map = Config::parse_model_map(
            "claude-3-5-sonnet=openai/gpt-5.2-chat\nclaude-haiku=openai/gpt-4.1-mini",
        )
        .unwrap();

        assert_eq!(
            model_map.get("claude-3-5-sonnet"),
            Some(&"openai/gpt-5.2-chat".to_string())
        );
        assert_eq!(
            model_map.get("claude-haiku"),
            Some(&"openai/gpt-4.1-mini".to_string())
        );
    }

    #[test]
    fn parse_model_map_rejects_invalid_entries() {
        let err = Config::parse_model_map("claude-3-5-sonnet").unwrap_err();

        assert!(err.to_string().contains("Expected source=target"));
    }

    #[test]
    fn parse_upstream_urls_splits_on_semicolons() {
        let urls = Config::parse_upstream_urls("https://openrouter.ai/api;https://api.openai.com")
            .unwrap();

        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://openrouter.ai/api");
        assert_eq!(urls[1], "https://api.openai.com");
    }

    #[test]
    fn parse_upstream_urls_single_url_still_works() {
        let urls = Config::parse_upstream_urls("https://api.openai.com").unwrap();
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn parse_upstream_urls_rejects_empty() {
        let err = Config::parse_upstream_urls("").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn parse_upstream_urls_validates_each_url() {
        let err = Config::parse_upstream_urls("https://api.openai.com;not-a-url").unwrap_err();
        assert!(err.to_string().contains("valid http"));
    }

    #[test]
    fn chat_completions_urls_resolves_all() {
        let config = Config {
            upstream_urls: vec![
                "https://openrouter.ai/api".to_string(),
                "https://api.openai.com".to_string(),
            ],
            ..Default::default()
        };

        let urls = config.chat_completions_urls();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://openrouter.ai/api/v1/chat/completions");
        assert_eq!(urls[1], "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn passthrough_api_key_defaults_to_false() {
        let config = Config::default();
        assert!(!config.passthrough_api_key);
    }

    #[test]
    fn passthrough_disabled_with_static_key_works() {
        let config = Config {
            api_key: Some("sk-test".to_string()),
            passthrough_api_key: false,
            ..Default::default()
        };
        assert!(!config.passthrough_api_key);
        assert_eq!(config.api_key, Some("sk-test".to_string()));
    }

    #[test]
    fn passthrough_enabled_with_no_static_key() {
        let config = Config {
            api_key: None,
            passthrough_api_key: true,
            ..Default::default()
        };
        assert!(config.passthrough_api_key);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn bind_defaults_to_zero_zero_zero_zero() {
        let config = Config::default();
        assert_eq!(config.bind, "0.0.0.0");
    }

    #[test]
    fn bind_accepts_loopback() {
        let config = Config {
            bind: "127.0.0.1".to_string(),
            ..Default::default()
        };
        assert_eq!(config.bind, "127.0.0.1");
    }
}
