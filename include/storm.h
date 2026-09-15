/*
 * StormLLM - Streaming Tensor Loader
 * 
 * A deterministic streaming execution engine with fixed memory pressure and zero stalls.
 * Treats model weights as a byte stream, processes like video frames, never rewinds, never caches.
 * 
 * Architecture:
 * - Chunk size: 64 MB (configurable)
 * - Active window: 6 chunks = 384 MB RAM (configurable)
 * - Prefetch queue: 5 deep (configurable)
 * - Total working set: ~704 MB in motion
 * 
 * Key Principles:
 * - Zero stall design: compute || load (parallel)
 * - No cache = no branching (no LRU, no hashmap, no reuse tracking)
 * - Sequential IO for max NVMe throughput
 * - Block-level dequantization (not chunk-level)
 * - Pointer rotation (no memcpy)
 */

#ifndef STORM_H
#define STORM_H

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>

/* ============================================================================
 * CONFIGURATION CONSTANTS
 * ============================================================================ */

#define DEFAULT_CHUNK_SIZE_MB       64
#define DEFAULT_ACTIVE_SLOTS        6
#define DEFAULT_PREFETCH_DEPTH      5
#define DEFAULT_TENSOR_RAM          2
#define DEFAULT_TENSOR_PREFETCH     2
#define MAX_PATH_LENGTH             512
#define MAX_TENSOR_NAME_LENGTH      256
#define MAX_TOOL_NAME_LENGTH        128

/* Memory slot states */
typedef enum {
    SLOT_EMPTY = 0,
    SLOT_LOADING,
    SLOT_READY,
    SLOT_ACTIVE,
    SLOT_EVICTED
} slot_state_t;

/* Quantization types supported */
typedef enum {
    QUANT_F32 = 0,
    QUANT_F16,
    QUANT_Q4_0,
    QUANT_Q4_1,
    QUANT_Q4_K,
    QUANT_Q5_0,
    QUANT_Q5_1,
    QUANT_Q5_K,
    QUANT_Q8_0,
    QUANT_UNKNOWN
} quant_type_t;

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * Chunk descriptor - represents a 64MB (or configured size) block of model data
 * This is the fundamental unit of streaming
 */
typedef struct {
    uint64_t file_offset;           /* Byte offset in model file */
    uint64_t size;                  /* Actual size in bytes (<= chunk_size) */
    uint64_t chunk_id;              /* Sequential chunk identifier */
    quant_type_t quant_type;        /* Quantization format */
    char tensor_name[MAX_TENSOR_NAME_LENGTH];  /* Associated tensor name */
    bool is_tail_chunk;             /* True if this is the last chunk of a tensor */
} chunk_desc_t;

/**
 * Memory slot - holds one chunk in RAM
 * Uses pointer rotation, not memory copying
 */
typedef struct {
    void* data;                     /* Pointer to chunk data in RAM */
    chunk_desc_t desc;              /* Chunk metadata */
    slot_state_t state;             /* Current state */
    size_t allocated_size;          /* Actual allocated memory size */
    pthread_mutex_t lock;           /* Slot-level lock for thread safety */
} memory_slot_t;

/**
 * Prefetch request - async load operation
 */
typedef struct {
    chunk_desc_t desc;              /* What to load */
    int slot_index;                 /* Target slot when ready */
    bool in_progress;               /* Currently being loaded */
    bool completed;                 /* Load finished */
    int error_code;                 /* Non-zero on failure */
} prefetch_req_t;

/**
 * GGUF Tensor metadata entry
 */
typedef struct {
    char name[MAX_TENSOR_NAME_LENGTH];
    uint64_t offset;                /* File offset */
    uint64_t size;                  /* Tensor size in bytes */
    uint32_t n_dims;                /* Number of dimensions */
    uint32_t dims[4];               /* Dimension sizes */
    quant_type_t quant_type;        /* Quantization type */
} tensor_meta_t;

/**
 * GGUF Model metadata
 */
typedef struct {
    char magic[4];                  /* GGUF magic number */
    uint32_t version;               /* GGUF version */
    uint64_t tensor_count;          /* Number of tensors */
    uint64_t kv_count;              /* Number of metadata key-values */
    tensor_meta_t* tensors;         /* Array of tensor metadata */
    uint64_t total_size;            /* Total model size in bytes */
    char model_path[MAX_PATH_LENGTH];
} gguf_metadata_t;

/**
 * Stream engine configuration
 */
typedef struct {
    /* Streaming parameters */
    size_t chunk_size_bytes;        /* Chunk size in bytes (default 64MB) */
    int active_slots;               /* Number of active RAM slots (default 6) */
    int prefetch_depth;             /* Prefetch queue depth (default 5) */
    
    /* Tensor-based streaming */
    bool stream_by_tensor;          /* Stream by tensor vs by MB chunk */
    int tensor_ram_limit;           /* Max tensors in RAM */
    int tensor_prefetch_count;      /* Tensors to prefetch */
    
    /* Paths */
    char model_path[MAX_PATH_LENGTH];
    char bridge_file[MAX_PATH_LENGTH];
    
    /* Debug */
    bool verbose;
} stream_config_t;

/**
 * Main streaming engine state
 * This is the core data structure that manages the entire pipeline
 */
typedef struct {
    /* Configuration */
    stream_config_t config;
    
    /* Model metadata */
    gguf_metadata_t metadata;
    int fd;                         /* Model file descriptor */
    
    /* Memory management */
    memory_slot_t* slots;           /* Array of active memory slots */
    int num_slots;                  /* Number of slots */
    int current_slot;               /* Currently executing slot index */
    
    /* Prefetch queue */
    prefetch_req_t* prefetch_queue; /* Circular buffer of prefetch requests */
    int prefetch_head;              /* Head of prefetch queue */
    int prefetch_tail;              /* Tail of prefetch queue */
    int prefetch_count;             /* Current items in queue */
    pthread_t prefetch_thread;      /* Background prefetch thread */
    pthread_mutex_t prefetch_lock;  /* Prefetch queue lock */
    pthread_cond_t prefetch_cond;   /* Signal new prefetch available */
    
    /* Execution state */
    uint64_t next_chunk_id;         /* Next chunk to schedule */
    uint64_t total_chunks;          /* Total chunks in model */
    bool running;                   /* Engine is active */
    bool shutdown;                  /* Shutdown requested */
    
    /* Bridge file (communication.bin) */
    FILE* bridge_fp;                /* Bridge file handle */
    uint64_t bridge_offset;         /* Current read/write position */
    
    /* Statistics */
    uint64_t bytes_loaded;
    uint64_t chunks_processed;
    uint64_t prefetch_hits;
    uint64_t prefetch_misses;
    uint64_t rotations_performed;
} stream_engine_t;

/**
 * Tool calling interface
 */
typedef struct {
    char name[MAX_TOOL_NAME_LENGTH];
    char allowed_directory[MAX_PATH_LENGTH];
    bool (*init)(void* ctx);
    int (*execute)(void* ctx, const char* command, char* output, size_t output_size);
    void (*cleanup)(void* ctx);
    void* context;
} tool_t;

/**
 * Filesystem tool context
 */
typedef struct {
    char base_directory[MAX_PATH_LENGTH];
    FILE* log_file;
    bool privileges_ask;            /* true=ask, false=allow */
} fs_tool_ctx_t;

/* ============================================================================
 * CORE API FUNCTIONS
 * ============================================================================ */

/**
 * Initialize the streaming engine
 * @param engine Pointer to engine structure to initialize
 * @param config Configuration parameters
 * @return 0 on success, negative error code on failure
 */
int stream_engine_init(stream_engine_t* engine, const stream_config_t* config);

/**
 * Load and parse GGUF metadata from model file
 * @param engine Engine instance
 * @return 0 on success, negative error code on failure
 */
int stream_engine_load_metadata(stream_engine_t* engine);

/**
 * Start the streaming engine (spawns prefetch thread)
 * @param engine Engine instance
 * @return 0 on success, negative error code on failure
 */
int stream_engine_start(stream_engine_t* engine);

/**
 * Get next chunk for processing (advances rotation)
 * @param engine Engine instance
 * @param slot_index Output: index of slot with next ready chunk
 * @return Pointer to memory_slot_t, NULL if nothing ready
 */
memory_slot_t* stream_engine_get_next_chunk(stream_engine_t* engine, int* slot_index);

/**
 * Mark current chunk as processed and trigger rotation
 * @param engine Engine instance
 * @param slot_index Slot index that was processed
 * @return 0 on success, negative error code on failure
 */
int stream_engine_complete_chunk(stream_engine_t* engine, int slot_index);

/**
 * Shutdown the streaming engine gracefully
 * @param engine Engine instance
 */
void stream_engine_shutdown(stream_engine_t* engine);

/**
 * Cleanup and free all resources
 * @param engine Engine instance
 */
void stream_engine_destroy(stream_engine_t* engine);

/* ============================================================================
 * DEQUANTIZATION API
 * ============================================================================ */

/**
 * Dequantize a single block (NOT entire chunk)
 * This is the critical path - must be fast and use minimal temp memory
 * 
 * @param quant_data Pointer to quantized block
 * @param quant_type Quantization type
 * @param out_buffer Output FP32 buffer (must be pre-allocated)
 * @param out_size Size of output buffer in elements
 * @return 0 on success, negative error code on failure
 */
int dequantize_block(const void* quant_data, quant_type_t quant_type, 
                     float* out_buffer, size_t out_size);

/**
 * Fused dequantize + dot product (AVX2/AVX512 optimized)
 * No intermediate buffer - computes directly into accumulator
 * 
 * @param quant_data Pointer to quantized block
 * @param quant_type Quantization type
 * @param weights Pointer to input weights/activations
 * @param result Accumulator for result
 * @param count Number of elements
 * @return 0 on success, negative error code on failure
 */
int dequantize_dot_product(const void* quant_data, quant_type_t quant_type,
                           const float* weights, float* result, size_t count);

/* ============================================================================
 * BRIDGE FILE API (communication.bin)
 * ============================================================================ */

/**
 * Open bridge file for reading/writing
 * @param engine Engine instance
 * @param path Path to bridge file
 * @return 0 on success, negative error code on failure
 */
int bridge_open(stream_engine_t* engine, const char* path);

/**
 * Write token data to bridge (from tokenizer)
 * @param engine Engine instance
 * @param data Binary token data
 * @param size Size in bytes
 * @return Bytes written, negative on error
 */
ssize_t bridge_write_token(stream_engine_t* engine, const void* data, size_t size);

/**
 * Read processed data from bridge (for transformer)
 * @param engine Engine instance
 * @param buffer Output buffer
 * @param size Buffer size
 * @return Bytes read, negative on error
 */
ssize_t bridge_read_processed(stream_engine_t* engine, void* buffer, size_t size);

/**
 * Close bridge file
 * @param engine Engine instance
 */
void bridge_close(stream_engine_t* engine);

/* ============================================================================
 * TOOL API
 * ============================================================================ */

/**
 * Initialize filesystem tool
 * @param ctx Tool context
 * @param base_dir Base directory (all operations confined here)
 * @param privileges_ask If true, ask before dangerous operations
 * @return 0 on success, negative error code on failure
 */
int tool_filesystem_init(fs_tool_ctx_t* ctx, const char* base_dir, bool privileges_ask);

/**
 * Execute filesystem command safely
 * Allowed commands: create_files, edit_files, write_files, read_files, 
 *                   list_directory, grep, compile_project
 * @param ctx Tool context
 * @param command Command string
 * @param output Output buffer
 * @param output_size Output buffer size
 * @return 0 on success, negative error code on failure
 */
int tool_filesystem_execute(fs_tool_ctx_t* ctx, const char* command, 
                            char* output, size_t output_size);

/**
 * Cleanup filesystem tool
 * @param ctx Tool context
 */
void tool_filesystem_cleanup(fs_tool_ctx_t* ctx);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * Parse INI configuration file
 * @param path Path to settings.ini
 * @param config Output configuration
 * @return 0 on success, negative error code on failure
 */
int config_parse_ini(const char* path, stream_config_t* config);

/**
 * Get quantization type from string
 * @param qtype_str String like "Q4_K", "F16", etc.
 * @return Quantization type enum
 */
quant_type_t quant_type_from_string(const char* qtype_str);

/**
 * Get human-readable quantization name
 * @param qtype Quantization type
 * @return Static string with name
 */
const char* quant_type_to_string(quant_type_t qtype);

/**
 * Calculate blocks per chunk for given quantization type
 * @param quant_type Quantization type
 * @param chunk_size Chunk size in bytes
 * @return Number of blocks
 */
size_t calc_blocks_per_chunk(quant_type_t quant_type, size_t chunk_size);

#endif /* STORM_H */
