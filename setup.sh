#!/bin/bash

# Print a message indicating starting installation
echo "Starting project setup..."

# Install poetry dependencies
echo "Installing poetry dependencies..."
poetry install

# Install flash-attn, skipping the CUDA build should be fine
echo "Installing flash-attn..."
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE poetry run python -m pip install flash-attn --no-build-isolation

# Print a completion message
echo "Setup complete."
