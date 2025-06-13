import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.esrgan.RRDBNet_arch import RRDBNet

def load_generator(ckpt_path, device):
    # ESRGAN generator definition: RRDBNet(3, 3, 64, 23, gc=32)
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.to(device).eval()
    return model

def upscale_image(model, image_path, save_path=None):
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB")
    # No need to resize; ESRGAN handles arbitrary LR sizes
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(input_tensor)
    sr_img = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)

    if save_path:
        Image.fromarray(sr_img).save(save_path)
        print(f"✅ Saved upscaled image to: {save_path}")

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    generator_ckpt = os.path.join("models", "esrgan", "generator.pth")
    input_image = "image.jpg"  # or your Zero-DCE enhanced + downsampled image
    output_image = "image_sr_output.jpg"
    model = load_generator(generator_ckpt, device)
    upscale_image(model, input_image, output_image)
