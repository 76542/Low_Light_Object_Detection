import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# 🔧 Add root directory to path so we can import from models/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from models.zero_dce.model import ZeroDCE

# --------------------------------------------
# Load trained Zero-DCE model from checkpoint
# --------------------------------------------
def load_model(ckpt_path, device):
    model = ZeroDCE()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model

# --------------------------------------------
# Enhance a single image (e.g., JPG/PNG)
# --------------------------------------------
def enhance_image(model, image_path, save_path=None):
    device = next(model.parameters()).device

    img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced, _ = model(input_tensor)

    # Convert output tensor to NumPy image
    enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
    enhanced_np = np.clip(enhanced_np, 0, 255).astype(np.uint8)

    # Save or show
    if save_path:
        Image.fromarray(enhanced_np).save(save_path)
        print(f"✅ Saved enhanced image to: {save_path}")
    else:
        cv2.imshow("Enhanced Image", cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --------------------------------------------
# Enhance a webcam stream or video file
# --------------------------------------------
def enhance_video(model, video_path=None):
    device = next(model.parameters()).device
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced, _ = model(tensor)

        out_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
        out_np = np.clip(out_np, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        cv2.imshow("Enhanced Frame", out_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------------------------------
# 🚀 Entry Point
# --------------------------------------------
if __name__ == "__main__":
    # Auto-select device (MPS for Mac, else CUDA or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your trained model
    model_path = os.path.join(ROOT_DIR, "checkpoints/zero_dce/zerodce_epoch_50.pth")
    model = load_model(model_path, device)

    # Enhance a test image
    input_img = os.path.join(ROOT_DIR, "test.png")
    output_img = os.path.join(ROOT_DIR, "enhanced_test.png")
    enhance_image(model, input_img, output_img)

    # Uncomment to test webcam or video
    # enhance_video(model)
    # enhance_video(model, "input_video.mp4")
