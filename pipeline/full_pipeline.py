import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pipeline.upscale import load_generator, upscale_image
from pipeline.enhance import load_model as load_dce_model
from models.zero_dce.infer_utils import process_lowlight_np

def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def upscale_image_from_array(model, img_np):
    """
    Converts a BGR NumPy image to tensor, passes it through ESRGAN, and returns upscaled NumPy image.
    """
    device = next(model.parameters()).device
    rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        sr_tensor = model(tensor)

    sr_img = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

def process_frame(frame, dce_model, esrgan_model):
    enhanced = process_lowlight_np(dce_model, frame)
    denoised_pre = denoise_image(enhanced)
    upscaled = upscale_image_from_array(esrgan_model, denoised_pre)
    final = denoise_image(upscaled)
    return final

def run_pipeline(input_path, output_path, yolo_model_path, device='cpu'):
    is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))

    # Load models
    dce_model_path = os.path.join("checkpoints", "zero_dce", "zerodce_epoch_50.pth")
    dce_model = load_dce_model(dce_model_path, device)
    esrgan_ckpt = os.path.join("models", "esrgan", "generator.pth")
    esrgan_model = load_generator(esrgan_ckpt, device)
    yolo_model = YOLO(yolo_model_path)

    if is_video:
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = process_frame(frame, dce_model, esrgan_model)

            results = yolo_model(processed, device=device, save=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = yolo_model.names[class_id]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(processed, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(processed, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            out.write(processed)
            frame_count += 1
            print(f"Processed frame {frame_count}", end='\r')

        cap.release()
        out.release()
        print(f"[✓] Video saved to: {output_path}")
    else:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_path}")
        processed = process_frame(img, dce_model, esrgan_model)
        cv2.imwrite(output_path, processed)
        print(f"[✓] Processed image saved to: {output_path}")
        yolo_model(processed, device=device, save=True)
        print("[✓] YOLOv8 detection complete.")

if __name__ == "__main__":
    run_pipeline(
        input_path="data/test.png",           # image or video input
        output_path="output/final_out.png",   # output file
        yolo_model_path="weights/yolov8s.pt",
        device="mps"  # or 'cuda' or 'cpu'
    )
