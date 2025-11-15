use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Application-specific errors
#[derive(Error, Debug)]
pub enum ProxyError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Request transformation error: {0}")]
    Transform(String),

    #[error("Upstream API error: {0}")]
    Upstream(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ProxyError::Config(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ProxyError::Transform(msg) => (StatusCode::BAD_REQUEST, msg),
            ProxyError::Upstream(msg) => (StatusCode::BAD_GATEWAY, msg),
            ProxyError::Serialization(err) => {
                (StatusCode::BAD_REQUEST, format!("JSON error: {}", err))
            }
            ProxyError::Http(err) => {
                (StatusCode::BAD_GATEWAY, format!("HTTP error: {}", err))
            }
            ProxyError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(json!({
            "error": {
                "type": "proxy_error",
                "message": error_message,
            }
        }));

        (status, body).into_response()
    }
}

/// Result type for proxy operations
pub type ProxyResult<T> = Result<T, ProxyError>;
