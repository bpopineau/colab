from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from torchvision.transforms import ToPILImage, ToTensor

from mirnet.model import MIRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run photo restoration pipeline")
    parser.add_argument(
        "--mirnet-weights",
        type=Path,
        default=Path("mirnet") / "weights" / "mirnet_finetuned.pth",
        help="Path to MIRNet weights",
    )
    parser.add_argument(
        "--esrgan-weights",
        type=Path,
        default=Path("realesrgan") / "weights" / "RealESRGAN_x4plus.pth",
        help="Path to Real-ESRGAN weights",
    )
    return parser.parse_args()


def run_inference(mirnet_weights: Path, esrgan_weights: Path) -> None:
    input_img = Image.open(Path("input") / "sample.jpg").convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mirnet = MIRNet().eval().to(device)
    state = torch.load(mirnet_weights, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    mirnet.load_state_dict(state)

    x = ToTensor()(input_img).unsqueeze(0).to(device)
    with torch.no_grad():
        y = mirnet(x).clamp(0, 1)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    stage1_img = ToPILImage()(y.squeeze().cpu())
    stage1_path = output_dir / "stage1_mirnet.png"
    stage1_img.save(stage1_path)

    model = RRDBNet(3, 3, 64, 23, 32, scale=4)
    up = RealESRGANer(
        scale=4, model_path=str(esrgan_weights), model=model, device=device
    )
    output, _ = up.enhance(np.array(stage1_img), outscale=4)
    Image.fromarray(output).save(output_dir / "final_output.png")


def main() -> None:
    args = parse_args()
    run_inference(args.mirnet_weights, args.esrgan_weights)


if __name__ == "__main__":
    main()
