import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from mirnet.model import MIRNet
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np


def main():
    input_img = Image.open("input/sample.jpg").convert("RGB")

    # Stage 1: MIRNet enhancement
    mirnet = MIRNet().eval()
    if torch.cuda.is_available():
        mirnet = mirnet.cuda()
    state = torch.load("mirnet/weights/mirnet_finetuned.pth", map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    mirnet.load_state_dict(state)
    x = ToTensor()(input_img).unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        y = mirnet(x).clamp(0, 1)
    stage1_img = ToPILImage()(y.squeeze().cpu())
    stage1_img.save("output/stage1_mirnet.png")

    # Stage 2: Real-ESRGAN upscale + denoise
    model = RRDBNet(3, 3, 64, 23, 32, scale=4)
    up = RealESRGANer(
        scale=4,
        model_path="realesrgan/weights/RealESRGAN_x4plus.pth",
        model=model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    output, _ = up.enhance(np.array(stage1_img), outscale=4)
    Image.fromarray(output).save("output/final_output.png")


if __name__ == "__main__":
    main()
