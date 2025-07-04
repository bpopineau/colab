#!/bin/bash
set -e

# Install system dependencies
sudo apt-get update
sudo apt-get install -y libcudnn8 aria2

# Install Python dependencies used in notebooks
pip install --upgrade torch torchvision
pip install realesrgan gfpgan basicsr
pip install "audio-separator[gpu]==0.32.0" demucs aria2 yt_dlp

echo "Setup complete."

