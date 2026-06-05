#!/bin/bash
# Clean build artifacts and cache files before git push

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Cleaning build artifacts ==="

# Remove build directory
if [ -d "build" ]; then
    rm -rf build/
    echo "[done] removed build/"
fi

# Remove build_out directory
if [ -d "build_out" ]; then
    rm -rf build_out/
    echo "[done] removed build_out/"
fi

# Remove output directory
if [ -d "output" ]; then
    rm -rf output/
    echo "[done] removed output/"
fi

# Remove .arts cache (cpp.db etc)
if [ -d ".arts/.cache" ]; then
    rm -rf .arts/.cache/
    echo "[done] removed .arts/.cache/"
fi

# Remove .arts entire directory if only contains cache
if [ -d ".arts" ]; then
    rm -rf .arts/
    echo "[done] removed .arts/"
fi

# Remove python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "[done] removed __pycache__ and *.pyc"

# Remove .o files
find . -name "*.o" -delete 2>/dev/null
echo "[done] removed *.o"

echo "=== Clean complete ==="