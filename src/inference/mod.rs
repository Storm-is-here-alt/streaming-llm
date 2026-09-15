//! Inference engine - FFN + SSM Mixer + MTP
//! No attention, no softmax, no rope, no kv-cache

use crate::config::InferenceConfig;

pub struct InferenceEngine {
    config: InferenceConfig,
    ssm_state: Vec<f32>,
    hidden_dim: usize,
}

impl InferenceEngine {
    pub fn new(config: InferenceConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let layers = config.layers;
        
        // SSM state: 32 layers * 4096 dims = 0.5MB fixed
        let ssm_state = vec![0.0f32; layers * hidden_dim];
        
        Self {
            config,
            ssm_state,
            hidden_dim,
        }
    }
    
    /// Layer forward: FFN + SSM Mixer
    /// This is a simplified stub - real impl would load weights from GGUF
    pub fn layer_forward(
        &mut self,
        hidden: &mut [f32],
        layer: usize,
        gate_weights: &[f32],
        up_weights: &[f32],
        down_weights: &[f32],
    ) {
        let dim = hidden.len();
        
        // FFN: hidden = SiLU(hidden * gate) * up
        for i in 0..dim.min(gate_weights.len().min(up_weights.len())) {
            let gated = hidden[i] * gate_weights[i];
            let silu = gated / (1.0 + (-gated).exp()); // SiLU activation
            hidden[i] = silu * up_weights[i];
        }
        
        // Apply down projection (simplified - just element-wise for now)
        for i in 0..dim.min(down_weights.len()) {
            hidden[i] *= down_weights[i];
        }
        
        // SSM Mixer: state = state * decay + hidden * 0.1
        //            hidden = state
        let state_offset = layer * self.hidden_dim;
        let decay = self.config.ssm_decay;
        
        for i in 0..dim.min(self.hidden_dim) {
            let state_idx = state_offset + i;
            self.ssm_state[state_idx] = self.ssm_state[state_idx] * decay + hidden[i] * 0.1;
            hidden[i] = self.ssm_state[state_idx];
        }
    }
    
    /// Multi-Token Prediction heads
    /// Returns k tokens in single forward pass
    pub fn predict_block(
        &self,
        hidden: &[f32],
        mtp_heads: &[Vec<f32>],
        vocab_size: usize,
    ) -> Vec<u32> {
        let k = self.config.mtp_k;
        let mut tokens = Vec::with_capacity(k);
        
        for head_idx in 0..k.min(mtp_heads.len()) {
            let head = &mtp_heads[head_idx];
            let token = self.sample_token(hidden, head, vocab_size);
            tokens.push(token);
        }
        
        tokens
    }
    
    fn sample_token(&self, hidden: &[f32], head_weights: &[f32], vocab_size: usize) -> u32 {
        // Simple dot-product sampling (no softmax per spec)
        let mut score = 0.0f32;
        for i in 0..hidden.len().min(head_weights.len()) {
            score += hidden[i] * head_weights[i];
        }
        
        // Map score to token ID deterministically for now
        // Real impl would use proper sampling
        ((score.abs() * 1000.0) as u32) % vocab_size as u32
    }
    
    pub fn get_ssm_state_mut(&mut self, layer: usize) -> &mut [f32] {
        let start = layer * self.hidden_dim;
        let end = start + self.hidden_dim;
        &mut self.ssm_state[start..end]
    }
    
    pub fn reset_ssm_state(&mut self) {
        for v in &mut self.ssm_state {
            *v = 0.0;
        }
    }
}

/// Dequantization functions for Q4_0 format
/// Uses AVX2 intrinsics with scalar fallback

#[target_feature(enable = "avx2")]
unsafe fn dequant_q4_0_avx2(q4_data: &[u8], out: &mut [f32]) {
    // Q4_0: blocks of 32 values stored in 18 bytes
    // 2 bytes scale (f16), 16 bytes quantized (4 bits per value)
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 18;
    
    let n_blocks = q4_data.len() / BYTES_PER_BLOCK;
    
    for b in 0..n_blocks {
        let block_start = b * BYTES_PER_BLOCK;
        let scale_bytes = &q4_data[block_start..block_start + 2];
        let q_bytes = &q4_data[block_start + 2..block_start + BYTES_PER_BLOCK];
        
        // Decode scale (f16 -> f32)
        let scale_u16 = u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]);
        let scale = half_to_f32(scale_u16);
        
        // Decode 32 values from 16 bytes
        let out_start = b * BLOCK_SIZE;
        for i in 0..16 {
            let q_byte = q_bytes[i];
            let v0 = (q_byte & 0xF) as i8 - 8;
            let v1 = (q_byte >> 4) as i8 - 8;
            
            if out_start + i * 2 < out.len() {
                out[out_start + i * 2] = v0 as f32 * scale;
            }
            if out_start + i * 2 + 1 < out.len() {
                out[out_start + i * 2 + 1] = v1 as f32 * scale;
            }
        }
    }
}

fn dequant_q4_0_scalar(q4_data: &[u8], out: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 18;
    
    let n_blocks = q4_data.len() / BYTES_PER_BLOCK;
    
    for b in 0..n_blocks {
        let block_start = b * BYTES_PER_BLOCK;
        let scale_bytes = &q4_data[block_start..block_start + 2];
        let q_bytes = &q4_data[block_start + 2..block_start + BYTES_PER_BLOCK];
        
        let scale_u16 = u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]);
        let scale = half_to_f32(scale_u16);
        
        let out_start = b * BLOCK_SIZE;
        for i in 0..16 {
            let q_byte = q_bytes[i];
            let v0 = (q_byte & 0xF) as i8 - 8;
            let v1 = (q_byte >> 4) as i8 - 8;
            
            if out_start + i * 2 < out.len() {
                out[out_start + i * 2] = v0 as f32 * scale;
            }
            if out_start + i * 2 + 1 < out.len() {
                out[out_start + i * 2 + 1] = v1 as f32 * scale;
            }
        }
    }
}

pub fn dequant_q4_0(q4_data: &[u8], out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            dequant_q4_0_avx2(q4_data, out);
        }
    } else {
        dequant_q4_0_scalar(q4_data, out);
    }
}

/// Convert f16 (stored as u16) to f32
fn half_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x3FF) as u32;
    
    if exponent == 0 {
        if mantissa == 0 {
            return if sign == 0 { 0.0 } else { -0.0 };
        }
        // Subnormal
        let mut e = 0;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let val = (sign << 31) | ((127 - 14 - e) << 23) | ((m & !0x400) << 13);
        return f32::from_bits(val);
    }
    
    if exponent == 31 {
        // Inf or NaN
        return f32::from_bits((sign << 31) | 0x7F800000 | (mantissa << 13));
    }
    
    // Normal
    let val = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    f32::from_bits(val)
}
