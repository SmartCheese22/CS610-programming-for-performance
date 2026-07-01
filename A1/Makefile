# Makefile for roll assignment (single problem)
# Assumes:
#   Source file: problem3-dir/220285.cpp
#   Produces binary: producer_consumer

CXX        := g++
ROLL       := 220285

# Paths / sources
PROB_DIR   := problem3-dir
PROB_SRC   := $(PROB_DIR)/$(ROLL).cpp

# Binary name
BIN        := producer_consumer

# Flags
CXXFLAGS   := -std=c++17 -O2 -pthread -Wall -Wextra
DEBUGFLAGS := -std=c++17 -O0 -g -pthread -Wall -Wextra

.PHONY: all debug run format clean help

all: $(BIN)

$(BIN): $(PROB_SRC)
	@echo "Building $(BIN) from $(PROB_SRC)"
	$(CXX) $(CXXFLAGS) -I$(PROB_DIR) -o $(BIN) $(PROB_SRC)

debug: $(PROB_SRC)
	$(CXX) $(DEBUGFLAGS) -I$(PROB_DIR) -o $(BIN)_dbg $(PROB_SRC)

# Usage: make run ARGS="/path/in T Lmin Lmax M /path/out"
run: $(BIN)
	@echo ">>> Running ./$(BIN) $(ARGS)"
	@./$(BIN) $(ARGS)

# Optional: format the source (requires clang-format)
format:
	-which clang-format >/dev/null 2>&1 && clang-format -i $(PROB_SRC) || (echo "clang-format not found; skipping"; exit 0)

clean:
	@rm -f $(BIN) $(BIN)_dbg

help:
	@echo "Makefile targets:"
	@echo "  make          -> build release binary ($(BIN))"
	@echo "  make debug    -> build debug binary ($(BIN)_dbg)"
	@echo "  make run ARGS=\"...\" -> run program with arguments"
	@echo "  make format   -> run clang-format on source (if installed)"
	@echo "  make clean    -> remove binaries"
