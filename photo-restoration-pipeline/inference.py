from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
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
    return parser.parse_args()


def run_inference(mirnet_weights: Path) -> None:
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
    output_img = ToPILImage()(y.squeeze().cpu())
    output_img.save(output_dir / "final_output.png")


def main() -> None:
    args = parse_args()
    run_inference(args.mirnet_weights)


if __name__ == "__main__":
    main()
