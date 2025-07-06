# Development Environment

## Base image
Ubuntu 22.04, Python 3.11, CUDA 12, NVIDIA A100 (16 GB).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

The requirements.txt pins torch>=2.1, torchvision>=0.16, basicsr-fixed>=1.4.3, plus other libs needed for MIRNet.
