#!/bin/bash

# Print a message indicating starting installation
echo "Starting project setup..."

# Install poetry dependencies
echo "Installing poetry dependencies..."
poetry install

# Install flash-attn
echo "Installing flash-attn..."
poetry run python -m pip install flash-attn --no-build-isolation

# Print a completion message
echo "Setup complete."
