//! WebUI module - serves static/index.html and handles WebSocket connections

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    #[serde(rename = "t")]
    pub t: String,
    #[serde(rename = "q", default)]
    pub q: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockResponse {
    #[serde(rename = "t")]
    pub t: String,
    #[serde(rename = "tokens")]
    pub tokens: Vec<String>,
    #[serde(rename = "audit", default)]
    pub audit: Vec<AuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub offset: u64,
    pub score: f32,
    pub bm25: f32,
    pub recency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRequest {
    #[serde(rename = "t")]
    pub t: String,
    #[serde(rename = "ids")]
    pub ids: Vec<usize>,
    #[serde(rename = "reward")]
    pub reward: f32,
}

pub struct AppState {
    // Placeholder for shared state with main engine
}

pub fn create_router() -> Router<Arc<RwLock<AppState>>> {
    Router::new()
        .route("/", get(root))
        .route("/ws", get(ws_handler))
        .nest_service("/static", tower_http::services::ServeDir::new("static"))
}

async fn root() -> impl IntoResponse {
    axum::response::Html(include_str!("../../static/index.html"))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    use axum::extract::ws::Message;
    
    loop {
        match socket.recv().await {
            Some(Ok(Message::Text(text))) => {
                if let Ok(req) = serde_json::from_str::<ChatRequest>(&text) {
                    match req.t.as_str() {
                        "chat" => {
                            // Process chat request through Bayesian engine
                            let response = BlockResponse {
                                t: "block".to_string(),
                                tokens: vec!["[Response generated]".to_string()],
                                audit: vec![],
                            };
                            
                            let _ = socket
                                .send(Message::Text(serde_json::to_string(&response).unwrap()))
                                .await;
                            
                            let _ = socket.send(Message::Text("{\"t\":\"done\"}".to_string())).await;
                        }
                        "feedback" => {
                            if let Ok(fb) = serde_json::from_str::<FeedbackRequest>(&text) {
                                println!("Feedback received for ids {:?}: reward={}", fb.ids, fb.reward);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some(Ok(Message::Close(_))) | None => break,
            Some(Err(_)) => break,
            _ => {}
        }
    }
}
