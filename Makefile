# StormLLM Makefile
# Streaming Tensor Loader with Deterministic Memory Pressure

CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -pthread
LDFLAGS = -pthread -lm

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRCS = $(SRC_DIR)/storm.c $(SRC_DIR)/dequant.c $(SRC_DIR)/tools.c $(SRC_DIR)/main.c
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/storm

# Include path
INCLUDES = -I$(INC_DIR)

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Link
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Built: $@"

# Compile
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Install (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/storm

# Debug build
debug: CFLAGS += -DDEBUG -g3 -O0
debug: clean all

# Release build
release: CFLAGS += -O3 -DNDEBUG
release: clean all

# Run demo
run: $(TARGET)
	./$(TARGET) -v

# Run with model (example)
# run-model: $(TARGET)
# 	./$(TARGET) -m /path/to/model.gguf -v

# Help
help:
	@echo "StormLLM Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build storm (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  debug    - Build with debug symbols and no optimization"
	@echo "  release  - Build with full optimization"
	@echo "  run      - Build and run demo"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Configuration:"
	@echo "  Edit config/settings.ini before running"

.PHONY: all clean install debug release run help directories
