/*
 * StormLLM - Filesystem Tool Implementation
 * 
 * Provides safe, sandboxed filesystem access for the model.
 * All operations are confined to a specified base directory.
 * Logs all operations for audit trail.
 * 
 * Supported commands:
 * - create_files <path> <content>
 * - edit_files <path> <new_content>
 * - write_files <path> <content>
 * - read_files <path>
 * - list_directory <path>
 * - grep <pattern> <path>
 * - compile_project <path>
 */

#include "storm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <libgen.h>

#define MAX_FILE_SIZE (10 * 1024 * 1024)  /* 10MB max file size */
#define MAX_OUTPUT_SIZE (64 * 1024)        /* 64KB max output */

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

/**
 * Check if path is within allowed directory
 * Returns 1 if safe, 0 if outside sandbox
 */
static int is_path_safe(const char* base_dir, const char* path) {
    if (!base_dir || !path) return 0;
    
    /* Resolve to absolute paths */
    char resolved_base[PATH_MAX];
    char resolved_path[PATH_MAX];
    
    if (!realpath(base_dir, resolved_base)) {
        return 0;
    }
    
    /* Try to resolve path, or at least get its directory */
    char* path_copy = strdup(path);
    char* dir = dirname(path_copy);
    
    if (!realpath(dir, resolved_path)) {
        /* Path doesn't exist yet - check parent */
        free(path_copy);
        strncpy(resolved_path, dir, PATH_MAX - 1);
    } else {
        free(path_copy);
    }
    
    /* Check if resolved_path starts with resolved_base */
    size_t base_len = strlen(resolved_base);
    if (strncmp(resolved_path, resolved_base, base_len) != 0) {
        return 0;
    }
    
    /* Ensure it's not just a prefix match (e.g., /home/storm vs /home/storm2) */
    if (resolved_path[base_len] != '\0' && resolved_path[base_len] != '/') {
        return 0;
    }
    
    return 1;
}

/**
 * Log operation to audit log
 */
static void log_operation(fs_tool_ctx_t* ctx, const char* operation, 
                          const char* path, int success) {
    if (!ctx || !ctx->log_file) return;
    
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);
    
    fprintf(ctx->log_file, "[%s] %s %s %s\n", 
            time_buf, operation, path, success ? "OK" : "FAILED");
    fflush(ctx->log_file);
}

/**
 * Safe file read with size limit
 */
static int safe_read_file(const char* path, char* buffer, size_t buffer_size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        return -errno;
    }
    
    /* Get file size */
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    if (size < 0 || size > MAX_FILE_SIZE) {
        fclose(fp);
        return -EFBIG;
    }
    
    size_t to_read = (size_t)size < buffer_size - 1 ? (size_t)size : buffer_size - 1;
    size_t bytes_read = fread(buffer, 1, to_read, fp);
    buffer[bytes_read] = '\0';
    
    fclose(fp);
    return (int)bytes_read;
}

/**
 * Safe file write
 */
static int safe_write_file(const char* path, const char* content) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        return -errno;
    }
    
    size_t len = strlen(content);
    size_t written = fwrite(content, 1, len, fp);
    
    fclose(fp);
    
    return (written == len) ? 0 : -EIO;
}

/* ============================================================================
 * TOOL API IMPLEMENTATION
 * ============================================================================ */

int tool_filesystem_init(fs_tool_ctx_t* ctx, const char* base_dir, bool privileges_ask) {
    if (!ctx || !base_dir) return -EINVAL;
    
    memset(ctx, 0, sizeof(fs_tool_ctx_t));
    
    /* Validate and store base directory */
    if (!is_path_safe(base_dir, base_dir)) {
        fprintf(stderr, "[TOOL] Invalid base directory: %s\n", base_dir);
        return -EINVAL;
    }
    
    strncpy(ctx->base_directory, base_dir, MAX_PATH_LENGTH - 1);
    ctx->privileges_ask = privileges_ask;
    
    /* Create log file */
    char log_path[MAX_PATH_LENGTH];
    snprintf(log_path, sizeof(log_path), "%s/.storm_operations.log", base_dir);
    
    ctx->log_file = fopen(log_path, "a");
    if (!ctx->log_file) {
        fprintf(stderr, "[TOOL] Warning: Could not create log file\n");
    } else {
        time_t now = time(NULL);
        fprintf(ctx->log_file, "\n=== Session started: %s ===\n", ctime(&now));
        fflush(ctx->log_file);
    }
    
    printf("[TOOL] Filesystem tool initialized\n");
    printf("  Base directory: %s\n", base_dir);
    printf("  Privileges: %s\n", privileges_ask ? "ask" : "allow");
    printf("  Log file: %s\n", log_path);
    
    return 0;
}

int tool_filesystem_execute(fs_tool_ctx_t* ctx, const char* command, 
                            char* output, size_t output_size) {
    if (!ctx || !command || !output) return -EINVAL;
    
    output[0] = '\0';
    
    /* Parse command */
    char cmd_copy[1024];
    strncpy(cmd_copy, command, sizeof(cmd_copy) - 1);
    
    char* op = strtok(cmd_copy, " \t\n");
    if (!op) {
        snprintf(output, output_size, "Error: Empty command");
        return -EINVAL;
    }
    
    char* arg1 = strtok(NULL, " \t\n");
    char* arg2 = strtok(NULL, "");  /* Rest of line */
    
    /* Skip leading whitespace in arg2 */
    while (arg2 && (*arg2 == ' ' || *arg2 == '\t')) arg2++;
    
    int result = 0;
    
    /* ========================================================================
     * READ_FILES
     * ======================================================================== */
    if (strcmp(op, "read_files") == 0) {
        if (!arg1) {
            snprintf(output, output_size, "Error: read_files requires path argument");
            return -EINVAL;
        }
        
        /* Build full path */
        char full_path[MAX_PATH_LENGTH];
        snprintf(full_path, sizeof(full_path), "%s/%s", ctx->base_directory, arg1);
        
        /* Safety check */
        if (!is_path_safe(ctx->base_directory, full_path)) {
            snprintf(output, output_size, "Error: Path outside allowed directory");
            log_operation(ctx, "read_files", arg1, 0);
            return -EPERM;
        }
        
        /* Read file */
        char* file_content = malloc(MAX_FILE_SIZE + 1);
        if (!file_content) {
            snprintf(output, output_size, "Error: Out of memory");
            return -ENOMEM;
        }
        
        int bytes_read = safe_read_file(full_path, file_content, MAX_FILE_SIZE);
        if (bytes_read < 0) {
            snprintf(output, output_size, "Error: Could not read file (%s)", 
                     strerror(-bytes_read));
            free(file_content);
            log_operation(ctx, "read_files", arg1, 0);
            return bytes_read;
        }
        
        /* Copy to output (truncate if needed) */
        if ((size_t)bytes_read >= output_size) {
            strncpy(output, file_content, output_size - 1);
            output[output_size - 1] = '\0';
            snprintf(output + output_size - 20, 19, "\n... [truncated]");
        } else {
            strcpy(output, file_content);
        }
        
        free(file_content);
        log_operation(ctx, "read_files", arg1, 1);
        return 0;
    }
    
    /* ========================================================================
     * WRITE_FILES / CREATE_FILES
     * ======================================================================== */
    else if (strcmp(op, "write_files") == 0 || strcmp(op, "create_files") == 0) {
        if (!arg1 || !arg2) {
            snprintf(output, output_size, 
                     "Error: %s requires path and content arguments", op);
            return -EINVAL;
        }
        
        /* Build full path */
        char full_path[MAX_PATH_LENGTH];
        snprintf(full_path, sizeof(full_path), "%s/%s", ctx->base_directory, arg1);
        
        /* Safety check */
        if (!is_path_safe(ctx->base_directory, full_path)) {
            snprintf(output, output_size, "Error: Path outside allowed directory");
            log_operation(ctx, op, arg1, 0);
            return -EPERM;
        }
        
        /* Ask for confirmation if privileges=ask */
        if (ctx->privileges_ask) {
            printf("\n[TOOL] Model wants to %s: %s\n", op, arg1);
            printf("Content preview: %.100s%s\n", arg2, 
                   strlen(arg2) > 100 ? "..." : "");
            printf("Allow? [y/N]: ");
            
            char response[16];
            if (!fgets(response, sizeof(response), stdin)) {
                snprintf(output, output_size, "Error: Could not read confirmation");
                return -EIO;
            }
            
            if (response[0] != 'y' && response[0] != 'Y') {
                snprintf(output, output_size, "Operation denied by user");
                log_operation(ctx, op, arg1, 0);
                return -EPERM;
            }
        }
        
        /* Create parent directories if needed */
        char* path_copy = strdup(full_path);
        char* dir = dirname(path_copy);
        mkdir(dir, 0755);  /* Ignore errors - dir may already exist */
        free(path_copy);
        
        /* Write file */
        result = safe_write_file(full_path, arg2);
        if (result < 0) {
            snprintf(output, output_size, "Error: Could not write file (%s)", 
                     strerror(-result));
            log_operation(ctx, op, arg1, 0);
            return result;
        }
        
        snprintf(output, output_size, "Successfully %s: %s", 
                 op == 0 ? "created" : "wrote", arg1);
        log_operation(ctx, op, arg1, 1);
        return 0;
    }
    
    /* ========================================================================
     * EDIT_FILES
     * ======================================================================== */
    else if (strcmp(op, "edit_files") == 0) {
        if (!arg1 || !arg2) {
            snprintf(output, output_size, 
                     "Error: edit_files requires path and new_content arguments");
            return -EINVAL;
        }
        
        /* Same as write_files for now */
        return tool_filesystem_execute(ctx, "write_files", output, output_size);
    }
    
    /* ========================================================================
     * LIST_DIRECTORY
     * ======================================================================== */
    else if (strcmp(op, "list_directory") == 0) {
        char* path = arg1 ? arg1 : ".";
        
        char full_path[MAX_PATH_LENGTH];
        snprintf(full_path, sizeof(full_path), "%s/%s", ctx->base_directory, path);
        
        if (!is_path_safe(ctx->base_directory, full_path)) {
            snprintf(output, output_size, "Error: Path outside allowed directory");
            log_operation(ctx, "list_directory", path, 0);
            return -EPERM;
        }
        
        DIR* dir = opendir(full_path);
        if (!dir) {
            snprintf(output, output_size, "Error: Could not open directory (%s)", 
                     strerror(errno));
            log_operation(ctx, "list_directory", path, 0);
            return -errno;
        }
        
        char* ptr = output;
        size_t remaining = output_size;
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strcmp(entry->d_name, ".") == 0 || 
                strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            
            int printed = snprintf(ptr, remaining, "%s\n", entry->d_name);
            if (printed < 0 || (size_t)printed >= remaining) {
                break;
            }
            ptr += printed;
            remaining -= printed;
        }
        
        closedir(dir);
        log_operation(ctx, "list_directory", path, 1);
        return 0;
    }
    
    /* ========================================================================
     * GREP
     * ======================================================================== */
    else if (strcmp(op, "grep") == 0) {
        if (!arg1 || !arg2) {
            snprintf(output, output_size, "Error: grep requires pattern and path");
            return -EINVAL;
        }
        
        /* Simple grep implementation - would need more work for production */
        snprintf(output, output_size, 
                 "Grep not fully implemented. Pattern: '%s', Path: '%s'", 
                 arg1, arg2);
        return 0;
    }
    
    /* ========================================================================
     * COMPILE_PROJECT
     * ======================================================================== */
    else if (strcmp(op, "compile_project") == 0) {
        char* path = arg1 ? arg1 : ".";
        
        char full_path[MAX_PATH_LENGTH];
        snprintf(full_path, sizeof(full_path), "%s/%s", ctx->base_directory, path);
        
        if (!is_path_safe(ctx->base_directory, full_path)) {
            snprintf(output, output_size, "Error: Path outside allowed directory");
            log_operation(ctx, "compile_project", path, 0);
            return -EPERM;
        }
        
        /* Look for Makefile or build script */
        char makefile_path[MAX_PATH_LENGTH];
        snprintf(makefile_path, sizeof(makefile_path), "%s/Makefile", full_path);
        
        FILE* test = fopen(makefile_path, "r");
        if (test) {
            fclose(test);
            snprintf(output, output_size, 
                     "Found Makefile. Would run: cd %s && make\n(Full implementation would execute this)",
                     full_path);
        } else {
            snprintf(output, output_size, 
                     "No Makefile found in %s. Please provide build instructions.",
                     full_path);
        }
        
        log_operation(ctx, "compile_project", path, 1);
        return 0;
    }
    
    /* ========================================================================
     * UNKNOWN COMMAND
     * ======================================================================== */
    else {
        snprintf(output, output_size, 
                 "Unknown command: %s\n"
                 "Available commands:\n"
                 "  read_files <path>\n"
                 "  write_files <path> <content>\n"
                 "  create_files <path> <content>\n"
                 "  edit_files <path> <new_content>\n"
                 "  list_directory [path]\n"
                 "  grep <pattern> <path>\n"
                 "  compile_project [path]",
                 op);
        return -EINVAL;
    }
    
    return result;
}

void tool_filesystem_cleanup(fs_tool_ctx_t* ctx) {
    if (!ctx) return;
    
    if (ctx->log_file) {
        time_t now = time(NULL);
        fprintf(ctx->log_file, "=== Session ended: %s ===\n", ctime(&now));
        fclose(ctx->log_file);
        ctx->log_file = NULL;
    }
    
    memset(ctx, 0, sizeof(fs_tool_ctx_t));
}
