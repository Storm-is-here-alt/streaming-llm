//! Bayesian engine - replaces attention mechanism
//! O(c*n) quickselect scoring for chunk retrieval

use crate::config::BayesianConfig;
use crate::memory::{DiskLog, IdxEntry};
use std::collections::HashMap;

#[derive(Debug)]
pub struct ScoredChunk {
    pub idx: usize,
    pub offset: u64,
    pub score: f32,
    pub bm25_score: f32,
    pub recency_score: f32,
    pub importance_score: f32,
    pub reward_score: f32,
}

pub struct BayesianEngine {
    config: BayesianConfig,
    now_ts: u64,
}

impl BayesianEngine {
    pub fn new(config: BayesianConfig) -> Self {
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self { config, now_ts }
    }
    
    pub fn select_chunks(&mut self, query: &str, disk_log: &DiskLog) -> Vec<ScoredChunk> {
        self.now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let query_tokens: Vec<&str> = query.split_whitespace().collect();
        if query_tokens.is_empty() {
            return Vec::new();
        }
        
        // Compute IDF for query tokens
        let idf = self.compute_idf(&query_tokens, disk_log.get_index());
        
        // Score all chunks
        let mut scored: Vec<ScoredChunk> = disk_log.get_index()
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                let (bm25, tf_map) = self.bm25_score(&query_tokens, entry, &idf);
                
                if bm25 < self.config.reject_threshold {
                    return None;
                }
                
                let recency = self.recency_score(entry.ts);
                let importance = entry.importance;
                let reward = entry.reward;
                
                let recency_log = recency.ln().max(-10.0);
                let importance_log = importance.ln().max(-10.0);
                
                let total_score = 
                    self.config.w_bm25 * bm25 +
                    self.config.w_recency * recency_log +
                    self.config.w_importance * importance_log +
                    self.config.w_reward * reward;
                
                if total_score >= self.config.accept_threshold {
                    Some(ScoredChunk {
                        idx,
                        offset: entry.offset,
                        score: total_score,
                        bm25_score: bm25,
                        recency_score: recency,
                        importance_score: importance,
                        reward_score: reward,
                    })
                } else {
                    None
                }
            })
            .collect();
        
        // QuickSelect top_k
        if scored.len() > self.config.top_k {
            self.quickselect_top_k(&mut scored, self.config.top_k);
            scored.truncate(self.config.top_k);
        }
        
        // Sort by score descending for final output
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        scored
    }
    
    fn compute_idf<'a>(&self, tokens: &[&'a str], index: &[IdxEntry]) -> HashMap<&'a str, f32> {
        let n_docs = index.len() as f32;
        let mut idf = HashMap::new();
        
        for token in tokens {
            let df = index.iter()
                .filter(|entry| {
                    // Would need to read actual text to check TF, use approximation
                    // For now, assume all tokens appear in some documents
                    false // Placeholder - would need actual text access
                })
                .count() as f32;
            
            // Smooth IDF
            let idf_val = ((n_docs + 1.0) / (df + 1.0)).ln() + 1.0;
            idf.insert(*token, idf_val);
        }
        
        idf
    }
    
    fn bm25_score(
        &self,
        query_tokens: &[&str],
        entry: &IdxEntry,
        idf: &HashMap<&str, f32>,
    ) -> (f32, HashMap<&str, u32>) {
        // BM25 formula: sum IDF(q) * TF(q,C) * (k1+1) / (TF + k1 * (1 - b + b * len/avglen))
        // Simplified: we don't have avg_len easily, use fixed normalization
        
        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;
        
        // Approximate document length from entry.len
        let doc_len = entry.len as f32;
        let avg_doc_len = 512.0; // Assume average chunk is ~512 bytes
        
        let mut total_score = 0.0;
        let mut tf_map = HashMap::new();
        
        // For a proper implementation, we'd read the actual text and count TF
        // Here we use a simplified model based on entry metadata
        for token in query_tokens {
            // Placeholder TF - in real impl, read text and count occurrences
            let tf = 1.0;
            tf_map.insert(*token, 1u32);
            
            let idf_val = idf.get(token).unwrap_or(&1.0);
            
            let numerator = tf * (k1 + 1.0);
            let denominator = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len);
            
            total_score += idf_val * numerator / denominator;
        }
        
        (total_score, tf_map)
    }
    
    fn recency_score(&self, ts: u64) -> f32 {
        let age_secs = (self.now_ts.saturating_sub(ts)) as f32;
        let half_life_secs = self.config.recency_half_life_h * 3600.0;
        
        // exp(-age / half_life * ln(2))
        (-age_secs / half_life_secs * std::f32::consts::LN_2).exp()
    }
    
    fn quickselect_top_k(&self, arr: &mut [ScoredChunk], k: usize) {
        if k == 0 || k >= arr.len() {
            return;
        }
        
        let mut left = 0;
        let mut right = arr.len();
        
        while left < right {
            let pivot_idx = self.partition(arr, left, right);
            
            if pivot_idx == k {
                return;
            } else if pivot_idx < k {
                left = pivot_idx + 1;
            } else {
                right = pivot_idx;
            }
        }
    }
    
    fn partition(&self, arr: &mut [ScoredChunk], left: usize, right: usize) -> usize {
        let pivot_idx = left + (right - left) / 2;
        let pivot_score = arr[pivot_idx].score;
        
        // Move pivot to end
        arr.swap(pivot_idx, right - 1);
        let mut store_idx = left;
        
        for i in left..right - 1 {
            if arr[i].score > pivot_score {
                arr.swap(i, store_idx);
                store_idx += 1;
            }
        }
        
        // Move pivot back
        arr.swap(store_idx, right - 1);
        store_idx
    }
    
    pub fn fetch_chunk_texts(
        &self,
        scored: &[ScoredChunk],
        disk_log: &mut DiskLog,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut texts = Vec::with_capacity(scored.len());
        
        for sc in scored {
            let (_, _, text) = disk_log.read_at(sc.offset)?;
            texts.push(text);
        }
        
        Ok(texts)
    }
}
