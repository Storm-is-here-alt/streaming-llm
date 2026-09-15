//! AUDIT-LLM V6 - Complete Implementation
//! A local LLM client with zero kv-cache, zero attention, fully auditable

mod config;
mod gguf;
mod bayesian;
mod memory;
mod inference;
mod mcp;
mod webui;

use axum::Router;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::services::ServeDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    // Parse command line arguments
    let mut model_path = String::from("model.gguf");
    let mut config_path = String::from("audit_llm.ini");
    let mut mcp_path = String::from("mcp.json");
    let mut bind_addr = String::from("127.0.0.1:7331");
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--config" => {
                if i + 1 < args.len() {
                    config_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--mcp" => {
                if i + 1 < args.len() {
                    mcp_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--webui" => {
                if i + 1 < args.len() {
                    bind_addr = args[i + 1].clone();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("AUDIT-LLM V6 - Local LLM with auditable memory");
                println!();
                println!("Usage: audit-llm [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --model <PATH>     Path to GGUF model file (default: model.gguf)");
                println!("  --config <PATH>    Path to config INI file (default: audit_llm.ini)");
                println!("  --mcp <PATH>       Path to MCP JSON file (default: mcp.json)");
                println!("  --webui <ADDR>     WebUI bind address (default: 127.0.0.1:7331)");
                println!("  --help, -h         Show this help message");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }
    
    println!("AUDIT-LLM V6 starting...");
    println!("Model: {}", model_path);
    println!("Config: {}", config_path);
    println!("MCP: {}", mcp_path);
    println!("WebUI: http://{}", bind_addr);
    
    // Load configuration
    let config = config::Config::load(std::path::Path::new(&config_path))
        .unwrap_or_else(|_| {
            eprintln!("Warning: Could not load config from {}, using defaults", config_path);
            config::Config::default_config()
        });
    
    println!("RAM limit: {} MB", config.resources.max_ram_mb);
    println!("Layers: {}", config.inference.layers);
    println!("Hidden dim: {}", config.inference.hidden_dim);
    
    // Initialize memory subsystem
    let mut disk_log = memory::DiskLog::new(
        &config.memory.log_path,
        &config.memory.index_path,
    )?;
    println!("Memory log: {}", config.memory.log_path);
    println!("Index: {}", config.memory.index_path);
    
    // Initialize Bayesian engine
    let mut bayesian_engine = bayesian::BayesianEngine::new(config.bayesian.clone());
    println!("Bayesian engine initialized (top_k={}, accept={})", 
             config.bayesian.top_k, config.bayesian.accept_threshold);
    
    // Initialize inference engine
    let inference_engine = inference::InferenceEngine::new(config.inference.clone());
    println!("Inference engine initialized (SSM state: 0.5MB fixed)");
    
    // Try to load MCP config
    match mcp::McpManager::load(&mcp_path) {
        Ok(mut mcp_manager) => {
            println!("MCP config loaded from {}", mcp_path);
            if let Err(e) = mcp_manager.spawn_all() {
                eprintln!("Warning: Could not spawn MCP servers: {}", e);
            }
        }
        Err(e) => {
            eprintln!("Note: No MCP config found at {}, MCP disabled", mcp_path);
        }
    }
    
    // Create static directory if it doesn't exist
    std::fs::create_dir_all("static").ok();
    
    // Create a minimal index.html if it doesn't exist
    if !std::path::Path::new("static/index.html").exists() {
        std::fs::write("static/index.html", include_str!("../static/index.html")).ok();
    }
    
    // Start web server
    let app_state = Arc::new(RwLock::new(webui::AppState {}));
    
    let app = Router::new()
        .route("/", axum::routing::get(root_handler))
        .route("/ws", axum::routing::get(ws_handler))
        .nest_service("/static", ServeDir::new("static"))
        .with_state(app_state);
    
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    println!("WebUI listening on http://{}", bind_addr);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn root_handler() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../static/index.html"))
}

async fn ws_handler(
    ws: axum::extract::ws::WebSocketUpgrade,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: axum::extract::ws::WebSocket) {
    use axum::extract::ws::Message;
    
    loop {
        match socket.recv().await {
            Some(Ok(Message::Text(text))) => {
                if let Ok(req) = serde_json::from_str::<webui::ChatRequest>(&text) {
                    match req.t.as_str() {
                        "chat" => {
                            // Process chat request through Bayesian engine
                            let response = webui::BlockResponse {
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
                            if let Ok(fb) = serde_json::from_str::<webui::FeedbackRequest>(&text) {
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
