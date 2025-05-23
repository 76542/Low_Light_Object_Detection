import os
import sys
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2

# Add root directory to import paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from models.srgan.generator import Generator

# ---------------------------------------
# Load the trained SRGAN generator model
# ---------------------------------------
def load_generator(ckpt_path, device):
    model = Generator()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model

# ---------------------------------------
# Upscale an image using SRGAN generator
# ---------------------------------------
# handles inference and saving the output 
def upscale_image(model, image_path, save_path=None):
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB") #loads the input image as RGB

    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device) #converts pytorch tensor in shape [1,3,H,W] for the model

    with torch.no_grad():
        sr_tensor = model(input_tensor) #runs image through generator 

    sr_img = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
    sr_img = np.clip(sr_img, 0, 255).astype(np.uint8)

    if save_path:
        Image.fromarray(sr_img).save(save_path)
        print(f"✅ Saved upscaled image to: {save_path}")
    else:
        cv2.imshow("Super-Resolved Image", cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ---------------------------------------
# Run from terminal
# ---------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path to trained generator
    generator_ckpt = os.path.join(ROOT_DIR, "checkpoints/srgan/generator_epoch_10.pth")

    # Input: enhanced image from Zero-DCE
    input_image = os.path.join(ROOT_DIR, "test_lr.png")

    # Output: super-resolved image
    output_image = os.path.join(ROOT_DIR, "sr_output.png")

    # Load model and upscale image
    model = load_generator(generator_ckpt, device)
    upscale_image(model, input_image, output_image)
