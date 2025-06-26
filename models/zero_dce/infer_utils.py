import torch
import numpy as np
import cv2

def process_lowlight_np(model, frame_bgr):
    """
    Enhances a BGR OpenCV frame using a Zero-DCE model and returns the enhanced BGR frame.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(rgb / 255.).permute(2, 0, 1).unsqueeze(0).float().to(next(model.parameters()).device)

    with torch.no_grad():
        enhanced, _ = model(img_tensor)

    out_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out_np = np.clip(out_np * 255, 0, 255).astype('uint8')
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
