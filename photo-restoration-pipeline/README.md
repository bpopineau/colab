# Photo Restoration Pipeline

This project provides a two-stage pipeline for restoring low-quality photographs.
It first enhances images using **MIRNet** and then upsamples them with
**Real-ESRGAN**.

## Usage

Install dependencies and clone the Real-ESRGAN repository:

```bash
pip install -r requirements.txt
# Clone Real-ESRGAN inside the `realesrgan/` directory if not already present
# git clone https://github.com/xinntao/Real-ESRGAN.git realesrgan
```

Run the inference script:

```bash
python inference.py
```

A minimal demonstration notebook is provided in `notebook.ipynb` for
Colab-based workflows.
