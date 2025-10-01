#!/bin/bash
set -e

# Build script for slackagent
# Handles tokenizers library dependency for ONNX Runtime builds

TOKENIZERS_VERSION="v1.23.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
TOKENIZERS_LIB="$LIB_DIR/libtokenizers.a"

# Check if tokenizers library exists
if [ ! -f "$TOKENIZERS_LIB" ]; then
    echo "Building tokenizers library..."

    # Create lib directory
    mkdir -p "$LIB_DIR"

    # Clone and build tokenizers
    TEMP_DIR=$(mktemp -d)
    git clone --depth 1 --branch "$TOKENIZERS_VERSION" https://github.com/daulet/tokenizers.git "$TEMP_DIR"

    cd "$TEMP_DIR"
    make build

    # Copy library to project (we're still in TEMP_DIR)
    cp libtokenizers.a "$LIB_DIR/"

    # Cleanup
    cd "$SCRIPT_DIR"
    rm -rf "$TEMP_DIR"

    echo "✓ Tokenizers library built and installed to $LIB_DIR"
else
    echo "✓ Tokenizers library already exists"
fi

# Build the project
echo "Building slackagent..."
cd "$SCRIPT_DIR"
CGO_LDFLAGS="-L$LIB_DIR" go build -tags ORT ./...

echo "✓ Build complete!"
