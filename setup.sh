#!/bin/bash
set -e

# Upgrade packaging tools
pip install --upgrade pip wheel

# Install system dependencies
sudo apt-get update
sudo apt-get install -y libcudnn8 aria2

# Install Python dependencies used in notebooks
pip install --upgrade torch torchvision
pip install gfpgan basicsr
pip install "audio-separator[gpu]==0.32.0" demucs aria2 yt_dlp

# Install repo requirements if available
if [ -f requirements.txt ]; then
    pip install -r requirements.txt --quiet --retries 3 --timeout 120
fi

echo "Setup complete."

