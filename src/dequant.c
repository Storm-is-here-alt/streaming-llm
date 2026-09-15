/*
 * StormLLM - Block-Level Dequantization
 * 
 * Implements fast, minimal-memory dequantization for GGUF quantization formats.
 * Key principle: Dequantize block-by-block, NOT chunk-by-chunk.
 * Never hold more than a few KB of FP data at once.
 * 
 * Block layouts follow llama.cpp conventions:
 * - Q4_0: 32 elements per block, 2 + 2*sizeof(float16) bytes
 * - Q4_K: 32 elements per block, 2 + sizeof(float16) + 8 bytes
 * - etc.
 */

#include "storm.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>

/* ============================================================================
 * BLOCK SIZE CONSTANTS (from llama.cpp)
 * ============================================================================ */

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK_K 256

typedef struct {
    uint16_t d;           /* Delta */
    uint8_t qs[QK4_0/2];  /* Quantized values (4-bit) */
} block_q4_0;

typedef struct {
    uint16_t d;           /* Delta */
    uint16_t m;           /* Min */
    uint8_t qs[QK4_1/2];  /* Quantized values (4-bit) */
} block_q4_1;

typedef struct {
    uint16_t d;           /* Delta */
    uint8_t qh[QK5_0/4];  /* High bits */
    uint8_t qs[QK5_0/2];  /* Low bits */
} block_q5_0;

typedef struct {
    uint16_t d;           /* Delta */
    uint16_t m;           /* Min */
    uint8_t qh[QK5_1/4];  /* High bits */
    uint8_t qs[QK5_1/2];  /* Low bits */
} block_q5_1;

typedef struct {
    uint16_t d;           /* Delta */
    int8_t qs[QK8_0];     /* Quantized values (8-bit) */
} block_q8_0;

/* Q4_K block structure */
typedef struct {
    uint16_t d;                      /* Super-block scale */
    uint16_t dmin;                   /* Super-block min scale */
    uint8_t scales[4];               /* 4 mini-scales */
    uint8_t q[QK_K/2];               /* Quantized values */
} block_q4_k;

/* Q5_K block structure */
typedef struct {
    uint16_t d;                      /* Super-block scale */
    uint16_t dmin;                   /* Super-block min scale */
    uint8_t scales[4];               /* 4 mini-scales */
    uint8_t qh[QK_K/8];              /* High bits */
    uint8_t q[QK_K/2];               /* Low bits */
} block_q5_k;

/* ============================================================================
 * HELPER MACROS
 * ============================================================================ */

/* Convert fp16 (stored as uint16_t) to float32 */
static inline float fp16_to_fp32(uint16_t h) {
    union {
        uint32_t i;
        float f;
    } u;
    
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    
    if (exp == 0) {
        /* Subnormal or zero */
        if (mant == 0) {
            u.i = sign << 31;
            return u.f;
        }
        /* Subnormal: convert to fp32 subnormal */
        exp = 0;
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp++;
        }
        mant &= 0x3ff;
        u.i = (sign << 31) | (exp << 23) | (mant << 13);
        /* Adjust exponent bias difference */
        float val = u.f;
        return sign ? -val : val * 0.00006103515625f; /* 2^-14 */
    } else if (exp == 31) {
        /* Infinity or NaN */
        u.i = (sign << 31) | 0x7f800000 | (mant << 13);
        return u.f;
    }
    
    /* Normal case: adjust exponent bias from 15 to 127 */
    u.i = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return u.f;
}

/* ============================================================================
 * DEQUANTIZATION FUNCTIONS
 * ============================================================================ */

int dequantize_block(const void* quant_data, quant_type_t quant_type,
                     float* out_buffer, size_t out_size) {
    if (!quant_data || !out_buffer) return -EINVAL;
    
    switch (quant_type) {
        case QUANT_F32: {
            /* No dequantization needed - just copy */
            const float* src = (const float*)quant_data;
            size_t count = out_size < QK4_0 ? out_size : QK4_0;
            memcpy(out_buffer, src, count * sizeof(float));
            return 0;
        }
        
        case QUANT_F16: {
            /* Convert fp16 to fp32 */
            const uint16_t* src = (const uint16_t*)quant_data;
            size_t count = out_size < QK4_0 ? out_size : QK4_0;
            for (size_t i = 0; i < count; i++) {
                out_buffer[i] = fp16_to_fp32(src[i]);
            }
            return 0;
        }
        
        case QUANT_Q4_0: {
            /* Q4_0: 32 elements per block */
            const block_q4_0* block = (const block_q4_0*)quant_data;
            
            float d = fp16_to_fp32(block->d);
            
            for (int i = 0; i < QK4_0 && i < (int)out_size; i++) {
                int idx = i / 2;
                int shift = (i % 2) * 4;
                int8_t val = (block->qs[idx] >> shift) & 0xF;
                /* Q4_0: value = d * (val - 8) */
                out_buffer[i] = d * (val - 8);
            }
            return 0;
        }
        
        case QUANT_Q4_1: {
            /* Q4_1: 32 elements per block */
            const block_q4_1* block = (const block_q4_1*)quant_data;
            
            float d = fp16_to_fp32(block->d);
            float m = fp16_to_fp32(block->m);
            
            for (int i = 0; i < QK4_1 && i < (int)out_size; i++) {
                int idx = i / 2;
                int shift = (i % 2) * 4;
                int8_t val = (block->qs[idx] >> shift) & 0xF;
                /* Q4_1: value = d * val + m */
                out_buffer[i] = d * val + m;
            }
            return 0;
        }
        
        case QUANT_Q5_0: {
            /* Q5_0: 32 elements per block */
            const block_q5_0* block = (const block_q5_0*)quant_data;
            
            float d = fp16_to_fp32(block->d);
            
            for (int i = 0; i < QK5_0 && i < (int)out_size; i++) {
                int idx = i / 2;
                int shift = (i % 2) * 4;
                int8_t val = (block->qs[idx] >> shift) & 0xF;
                
                /* Add high bit */
                int8_t high_bit = (block->qh[i / 8] >> (i % 8)) & 0x1;
                val |= (high_bit << 4);
                
                /* Q5_0: value = d * (val - 16) */
                out_buffer[i] = d * (val - 16);
            }
            return 0;
        }
        
        case QUANT_Q5_1: {
            /* Q5_1: 32 elements per block */
            const block_q5_1* block = (const block_q5_1*)quant_data;
            
            float d = fp16_to_fp32(block->d);
            float m = fp16_to_fp32(block->m);
            
            for (int i = 0; i < QK5_1 && i < (int)out_size; i++) {
                int idx = i / 2;
                int shift = (i % 2) * 4;
                int8_t val = (block->qs[idx] >> shift) & 0xF;
                
                /* Add high bit */
                int8_t high_bit = (block->qh[i / 8] >> (i % 8)) & 0x1;
                val |= (high_bit << 4);
                
                /* Q5_1: value = d * val + m */
                out_buffer[i] = d * val + m;
            }
            return 0;
        }
        
        case QUANT_Q8_0: {
            /* Q8_0: 32 elements per block */
            const block_q8_0* block = (const block_q8_0*)quant_data;
            
            float d = fp16_to_fp32(block->d);
            
            for (int i = 0; i < QK8_0 && i < (int)out_size; i++) {
                /* Q8_0: value = d * val */
                out_buffer[i] = d * block->qs[i];
            }
            return 0;
        }
        
        case QUANT_Q4_K: {
            /* Q4_K: More complex super-block structure */
            const block_q4_k* block = (const block_q4_k*)quant_data;
            
            float d = fp16_to_fp32(block->d);
            
            /* Decode 4 mini-scales */
            float scales[4];
            for (int i = 0; i < 4; i++) {
                scales[i] = (block->scales[i] & 0xF) * d;
            }
            
            for (int i = 0; i < QK_K && i < (int)out_size; i++) {
                int idx = i / 2;
                int shift = (i % 2) * 4;
                int8_t val = (block->q[idx] >> shift) & 0xF;
                
                /* Apply appropriate scale based on position */
                int scale_idx = i / (QK_K / 4);
                out_buffer[i] = scales[scale_idx] * val;
            }
            return 0;
        }
        
        case QUANT_Q5_K: {
            /* Q5_K: Similar to Q4_K but with 5-bit values */
            const block_q5_k* block = (const block_q5_k*)quant_data;
            
            float d = fp16_to_fp32(block->d);
            
            /* Decode 4 mini-scales */
            float scales[4];
            for (int i = 0; i < 4; i++) {
                scales[i] = (block->scales[i] & 0xF) * d;
            }
            
            for (int i = 0; i < QK_K && i < (int)out_size; i++) {
                int idx = i / 2;
                int shift = (i % 2) * 4;
                int8_t val = (block->q[idx] >> shift) & 0xF;
                
                /* Add high bit */
                int8_t high_bit = (block->qh[i / 8] >> (i % 8)) & 0x1;
                val |= (high_bit << 4);
                
                /* Apply appropriate scale */
                int scale_idx = i / (QK_K / 4);
                out_buffer[i] = scales[scale_idx] * val;
            }
            return 0;
        }
        
        default:
            return -EINVAL;
    }
}

/* ============================================================================
 * FUSED DEQUANTIZE + DOT PRODUCT
 * For AVX2/AVX512 optimization, these would use SIMD intrinsics
 * This is a scalar reference implementation
 * ============================================================================ */

int dequantize_dot_product(const void* quant_data, quant_type_t quant_type,
                           const float* weights, float* result, size_t count) {
    if (!quant_data || !weights || !result) return -EINVAL;
    
    float acc = 0.0f;
    
    switch (quant_type) {
        case QUANT_Q4_0: {
            const block_q4_0* block = (const block_q4_0*)quant_data;
            float d = fp16_to_fp32(block->d);
            
            size_t blocks = (count + QK4_0 - 1) / QK4_0;
            
            for (size_t b = 0; b < blocks; b++) {
                for (int i = 0; i < QK4_0 && (b * QK4_0 + i) < count; i++) {
                    int idx = i / 2;
                    int shift = (i % 2) * 4;
                    int8_t val = (block->qs[idx] >> shift) & 0xF;
                    float weight = d * (val - 8);
                    
                    acc += weight * weights[b * QK4_0 + i];
                }
            }
            break;
        }
        
        case QUANT_Q4_1: {
            const block_q4_1* block = (const block_q4_1*)quant_data;
            float d = fp16_to_fp32(block->d);
            float m = fp16_to_fp32(block->m);
            
            size_t blocks = (count + QK4_1 - 1) / QK4_1;
            
            for (size_t b = 0; b < blocks; b++) {
                for (int i = 0; i < QK4_1 && (b * QK4_1 + i) < count; i++) {
                    int idx = i / 2;
                    int shift = (i % 2) * 4;
                    int8_t val = (block->qs[idx] >> shift) & 0xF;
                    float weight = d * val + m;
                    
                    acc += weight * weights[b * QK4_1 + i];
                }
            }
            break;
        }
        
        case QUANT_Q8_0: {
            const block_q8_0* block = (const block_q8_0*)quant_data;
            float d = fp16_to_fp32(block->d);
            
            size_t blocks = (count + QK8_0 - 1) / QK8_0;
            
            for (size_t b = 0; b < blocks; b++) {
                for (int i = 0; i < QK8_0 && (b * QK8_0 + i) < count; i++) {
                    float weight = d * block->qs[i];
                    acc += weight * weights[b * QK8_0 + i];
                }
            }
            break;
        }
        
        default: {
            /* Fallback: dequantize to temp buffer then dot product */
            float temp[QK_K];  /* Max block size */
            
            size_t processed = 0;
            while (processed < count) {
                size_t remaining = count - processed;
                size_t block_count = (remaining > QK_K) ? QK_K : remaining;
                
                if (dequantize_block((const uint8_t*)quant_data + 
                                     processed * sizeof(float), 
                                     quant_type, temp, block_count) != 0) {
                    return -EINVAL;
                }
                
                for (size_t i = 0; i < block_count; i++) {
                    acc += temp[i] * weights[processed + i];
                }
                
                processed += block_count;
            }
            break;
        }
    }
    
    *result = acc;
    return 0;
}
