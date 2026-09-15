//! MCP Manager - reads mcp.json and spawns child processes
//! Exact format as specified: mcpServers key with command/args/env

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub mcpServers: HashMap<String, McpServer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServer {
    pub command: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

pub struct McpManager {
    config: McpConfig,
    children: HashMap<String, Child>,
}

impl McpManager {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: McpConfig = serde_json::from_str(&content)?;
        
        Ok(Self {
            config,
            children: HashMap::new(),
        })
    }
    
    pub fn spawn_all(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for (name, server) in &self.config.mcpServers {
            self.spawn_server(name, server)?;
        }
        Ok(())
    }
    
    fn spawn_server(&mut self, name: &str, server: &McpServer) -> Result<(), Box<dyn std::error::Error>> {
        let mut cmd = Command::new(&server.command);
        cmd.args(&server.args);
        cmd.envs(&server.env);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::null());
        
        let child = cmd.spawn()?;
        self.children.insert(name.to_string(), child);
        
        Ok(())
    }
    
    pub fn call_tool(
        &mut self,
        server_name: &str,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        if let Some(child) = self.children.get_mut(server_name) {
            // Build JSON-RPC request
            let request = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            });
            
            // Write to stdin
            if let Some(mut stdin) = child.stdin.take() {
                let req_str = request.to_string();
                stdin.write_all(req_str.as_bytes())?;
                stdin.write_all(b"\n")?;
                stdin.flush()?;
                
                // Put stdin back
                child.stdin = Some(stdin);
            }
            
            // Read from stdout
            if let Some(mut stdout) = child.stdout.take() {
                let mut response = String::new();
                stdout.read_to_string(&mut response)?;
                
                // Put stdout back
                child.stdout = Some(stdout);
                
                // Parse response
                let parsed: serde_json::Value = serde_json::from_str(&response)?;
                return Ok(parsed);
            }
        }
        
        Err(format!("Server {} not found", server_name).into())
    }
    
    pub fn get_status(&self) -> HashMap<String, bool> {
        let mut status = HashMap::new();
        
        for (name, _) in &self.config.mcpServers {
            // Check if child is still running
            let running = self.children.get(name)
                .and_then(|child| child.try_get_exit_status().ok())
                .map(|s| !s.code().is_some())
                .unwrap_or(false);
            status.insert(name.clone(), running);
        }
        
        status
    }
    
    pub fn restart_server(&mut self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Kill existing
        if let Some(mut child) = self.children.remove(name) {
            let _ = child.kill();
            let _ = child.wait();
        }
        
        // Respawn
        if let Some(server) = self.config.mcpServers.get(name) {
            self.spawn_server(name, server)?;
        }
        
        Ok(())
    }
}
