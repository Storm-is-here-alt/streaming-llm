/*
 * StormLLM - Main Entry Point and Demo
 * 
 * Demonstrates the streaming tensor loader architecture.
 * 
 * Usage: ./storm [config_file]
 * 
 * Without arguments, uses default settings from config/settings.ini
 */

#include "storm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

/* Global engine for signal handler */
static stream_engine_t* g_engine = NULL;

/* Signal handler for graceful shutdown */
static void signal_handler(int sig) {
    if (g_engine) {
        fprintf(stderr, "\n[STORM] Received signal %d, shutting down...\n", sig);
        stream_engine_shutdown(g_engine);
    }
}

/* Print usage information */
static void print_usage(const char* program) {
    printf("StormLLM - Streaming Tensor Loader\n");
    printf("\n");
    printf("Usage: %s [options]\n", program);
    printf("\n");
    printf("Options:\n");
    printf("  -c, --config <file>   Path to settings.ini (default: config/settings.ini)\n");
    printf("  -m, --model <path>    Path to GGUF model file\n");
    printf("  -v, --verbose         Enable verbose logging\n");
    printf("  -h, --help            Show this help message\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s -c config/settings.ini -m /models/llama-7b.Q4_K.gguf\n", program);
    printf("\n");
}

/* Run a simple test/demonstration */
static int run_demo(stream_engine_t* engine) {
    printf("\n=== StormLLM Demo ===\n\n");
    
    printf("Configuration:\n");
    printf("  Chunk size: %zu MB\n", engine->config.chunk_size_bytes / (1024 * 1024));
    printf("  Active slots: %d\n", engine->num_slots);
    printf("  Prefetch depth: %d\n", engine->config.prefetch_depth);
    printf("  Total chunks: %lu\n", engine->total_chunks);
    printf("  Model size: %lu bytes\n", engine->metadata.total_size);
    printf("\n");
    
    /* Simulate processing loop */
    printf("Starting streaming simulation...\n\n");
    
    int processed = 0;
    int max_to_process = 10;  /* Demo: process only first 10 chunks */
    
    while (processed < max_to_process && engine->running) {
        int slot_index;
        memory_slot_t* chunk = stream_engine_get_next_chunk(engine, &slot_index);
        
        if (!chunk) {
            /* No chunk ready yet - wait a bit */
            usleep(10000);  /* 10ms */
            continue;
        }
        
        /* Process chunk (in real implementation, this would dequantize + compute) */
        printf("[DEMO] Processing chunk %lu in slot %d\n", 
               chunk->desc.chunk_id, slot_index);
        printf("       Offset: %lu, Size: %lu bytes\n", 
               chunk->desc.file_offset, chunk->desc.size);
        
        /* Simulate computation time */
        usleep(50000);  /* 50ms simulated compute */
        
        /* Mark chunk as complete (triggers rotation) */
        stream_engine_complete_chunk(engine, slot_index);
        
        processed++;
    }
    
    printf("\nDemo complete. Processed %d chunks.\n", processed);
    printf("\nStatistics:\n");
    printf("  Bytes loaded: %lu\n", engine->bytes_loaded);
    printf("  Chunks processed: %lu\n", engine->chunks_processed);
    printf("  Rotations performed: %lu\n", engine->rotations_performed);
    printf("  Prefetch hits: %lu\n", engine->prefetch_hits);
    printf("  Prefetch misses: %lu\n", engine->prefetch_misses);
    
    return 0;
}

int main(int argc, char* argv[]) {
    const char* config_path = "config/settings.ini";
    const char* model_path = NULL;
    bool verbose = false;
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) {
            if (i + 1 < argc) {
                config_path = argv[++i];
            } else {
                fprintf(stderr, "Error: --config requires an argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                fprintf(stderr, "Error: --model requires an argument\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    printf("=== StormLLM - Streaming Tensor Loader ===\n\n");
    
    /* Load configuration */
    stream_config_t config;
    memset(&config, 0, sizeof(config));
    
    printf("Loading configuration from: %s\n", config_path);
    if (config_parse_ini(config_path, &config) != 0) {
        fprintf(stderr, "Warning: Could not load config file, using defaults\n");
        
        /* Set minimal defaults */
        config.chunk_size_bytes = 64 * 1024 * 1024;  /* 64MB */
        config.active_slots = 6;
        config.prefetch_depth = 5;
        config.verbose = verbose;
    }
    
    /* Override with command line options */
    if (model_path) {
        strncpy(config.model_path, model_path, MAX_PATH_LENGTH - 1);
    }
    config.verbose = verbose || config.verbose;
    
    /* Initialize engine */
    stream_engine_t engine;
    g_engine = &engine;
    
    printf("Initializing streaming engine...\n");
    if (stream_engine_init(&engine, &config) != 0) {
        fprintf(stderr, "Error: Failed to initialize engine\n");
        return 1;
    }
    
    /* Setup signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Load metadata if model path specified */
    if (config.model_path[0]) {
        printf("Loading model metadata from: %s\n", config.model_path);
        if (stream_engine_load_metadata(&engine) != 0) {
            fprintf(stderr, "Error: Failed to load model metadata\n");
            stream_engine_destroy(&engine);
            return 1;
        }
        
        /* Open bridge file if configured */
        if (config.bridge_file[0]) {
            printf("Opening bridge file: %s\n", config.bridge_file);
            if (bridge_open(&engine, config.bridge_file) != 0) {
                fprintf(stderr, "Warning: Could not open bridge file\n");
            }
        }
        
        /* Start engine */
        printf("Starting streaming engine...\n");
        if (stream_engine_start(&engine) != 0) {
            fprintf(stderr, "Error: Failed to start engine\n");
            stream_engine_destroy(&engine);
            return 1;
        }
        
        /* Run demo/test */
        run_demo(&engine);
        
        /* Cleanup */
        bridge_close(&engine);
    } else {
        printf("\nNo model specified. Running in demo mode.\n");
        printf("Use -m or --model to specify a GGUF model file.\n");
        
        /* Create fake metadata for demo */
        engine.metadata.total_size = 4UL * 1024 * 1024 * 1024;  /* 4GB fake model */
        engine.total_chunks = engine.metadata.total_size / config.chunk_size_bytes;
        engine.running = true;
        
        /* Start prefetch thread for demo */
        stream_engine_start(&engine);
        
        run_demo(&engine);
    }
    
    /* Shutdown */
    printf("\nShutting down...\n");
    stream_engine_shutdown(&engine);
    stream_engine_destroy(&engine);
    
    g_engine = NULL;
    
    printf("Done.\n");
    return 0;
}
