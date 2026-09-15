//! Configuration module for audit-llm
//! Handles audit_llm.ini parsing and RAM governor settings

use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub resources: ResourcesConfig,
    pub memory: MemoryConfig,
    pub bayesian: BayesianConfig,
    pub inference: InferenceConfig,
    pub healing: HealingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourcesConfig {
    pub max_ram_mb: usize,
    pub weight_window_mb: usize,
    pub threads: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    pub log_path: String,
    pub index_path: String,
    pub chunk_tokens: usize,
    pub never_prune: bool,
    pub never_truncate: bool,
    pub max_feed_tokens: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BayesianConfig {
    pub top_k: usize,
    pub accept_threshold: f32,
    pub reject_threshold: f32,
    pub recency_half_life_h: f32,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub w_bm25: f32,
    pub w_recency: f32,
    pub w_importance: f32,
    pub w_reward: f32,
    pub audit_enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    pub layers: usize,
    pub hidden_dim: usize,
    pub mtp_k: usize,
    pub mtp_verify: bool,
    pub moe_top_n: usize,
    pub ssm_decay: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealingConfig {
    pub learning_rate: f32,
    pub auto_tune: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = ini::Ini::load_from_str(&content)?
            .into_section(None)
            .ok_or("No default section in ini")?;
        
        // Manual parsing since ini crate doesn't directly deserialize
        Self::parse_ini(&content)
    }
    
    fn parse_ini(content: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut map = std::collections::HashMap::new();
        let mut current_section = String::new();
        
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            
            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len()-1].to_string();
                continue;
            }
            
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                map.insert(format!("{}.{}", current_section, key), value.to_string());
            }
        }
        
        let get = |section: &str, key: &str| -> Option<&String> {
            map.get(&format!("{}.{}", section, key))
        };
        
        let parse_bool = |section: &str, key: &str, default: bool| -> bool {
            get(section, key).map(|s| s.parse().unwrap_or(default)).unwrap_or(default)
        };
        
        let parse_f32 = |section: &str, key: &str, default: f32| -> f32 {
            get(section, key).map(|s| s.parse().unwrap_or(default)).unwrap_or(default)
        };
        
        let parse_usize = |section: &str, key: &str, default: usize| -> usize {
            get(section, key).map(|s| s.parse().unwrap_or(default)).unwrap_or(default)
        };
        
        let parse_string = |section: &str, key: &str, default: &str| -> String {
            get(section, key).cloned().unwrap_or_else(|| default.to_string())
        };
        
        Ok(Config {
            resources: ResourcesConfig {
                max_ram_mb: parse_usize("resources", "max_ram_mb", 512),
                weight_window_mb: parse_usize("resources", "weight_window_mb", 64),
                threads: parse_usize("resources", "threads", 4),
            },
            memory: MemoryConfig {
                log_path: parse_string("memory", "log_path", "./chat.dat"),
                index_path: parse_string("memory", "index_path", "./chat.idx"),
                chunk_tokens: parse_usize("memory", "chunk_tokens", 128),
                never_prune: parse_bool("memory", "never_prune", true),
                never_truncate: parse_bool("memory", "never_truncate", true),
                max_feed_tokens: parse_usize("memory", "max_feed_tokens", 4096),
            },
            bayesian: BayesianConfig {
                top_k: parse_usize("bayesian", "top_k", 32),
                accept_threshold: parse_f32("bayesian", "accept_threshold", 0.35),
                reject_threshold: parse_f32("bayesian", "reject_threshold", 0.12),
                recency_half_life_h: parse_f32("bayesian", "recency_half_life_h", 24.0),
                bm25_k1: parse_f32("bayesian", "bm25_k1", 1.2),
                bm25_b: parse_f32("bayesian", "bm25_b", 0.75),
                w_bm25: parse_f32("bayesian", "w_bm25", 0.5),
                w_recency: parse_f32("bayesian", "w_recency", 0.2),
                w_importance: parse_f32("bayesian", "w_importance", 0.2),
                w_reward: parse_f32("bayesian", "w_reward", 0.1),
                audit_enabled: parse_bool("bayesian", "audit_enabled", true),
            },
            inference: InferenceConfig {
                layers: parse_usize("inference", "layers", 32),
                hidden_dim: parse_usize("inference", "hidden_dim", 4096),
                mtp_k: parse_usize("inference", "mtp_k", 4),
                mtp_verify: parse_bool("inference", "mtp_verify", true),
                moe_top_n: parse_usize("inference", "moe_top_n", 2),
                ssm_decay: parse_f32("inference", "ssm_decay", 0.9),
            },
            healing: HealingConfig {
                learning_rate: parse_f32("healing", "learning_rate", 0.05),
                auto_tune: parse_bool("healing", "auto_tune", true),
            },
        })
    }
    
    pub fn default_config() -> Self {
        Config {
            resources: ResourcesConfig {
                max_ram_mb: 512,
                weight_window_mb: 64,
                threads: 4,
            },
            memory: MemoryConfig {
                log_path: "./chat.dat".to_string(),
                index_path: "./chat.idx".to_string(),
                chunk_tokens: 128,
                never_prune: true,
                never_truncate: true,
                max_feed_tokens: 4096,
            },
            bayesian: BayesianConfig {
                top_k: 32,
                accept_threshold: 0.35,
                reject_threshold: 0.12,
                recency_half_life_h: 24.0,
                bm25_k1: 1.2,
                bm25_b: 0.75,
                w_bm25: 0.5,
                w_recency: 0.2,
                w_importance: 0.2,
                w_reward: 0.1,
                audit_enabled: true,
            },
            inference: InferenceConfig {
                layers: 32,
                hidden_dim: 4096,
                mtp_k: 4,
                mtp_verify: true,
                moe_top_n: 2,
                ssm_decay: 0.9,
            },
            healing: HealingConfig {
                learning_rate: 0.05,
                auto_tune: true,
            },
        }
    }
}
