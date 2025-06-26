import os
import cv2
import torch
from ultralytics import YOLO


def run_yolo_person_detection(model_path, image_path, save_path, device='cpu'):
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    print(f"Running inference on: {image_path}")
    results = model(image_path, device=device, save=False)

    for r in results:
        img = r.orig_img.copy()
        boxes = r.boxes
        drawn = False
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[class_id]
            if label.lower() == "person":
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(img, (xyxy[0], xyxy[1]),
                              (xyxy[2], xyxy[3]), (0, 0, 255), 3)
                text = f"{label} {conf:.2f}"
                cv2.putText(img, text, (xyxy[0], xyxy[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print(f"Box drawn for: {label} (confidence {conf:.2f}) at {xyxy}")
                drawn = True

        # Make sure the directory of the save_path exists
        save_folder = os.path.dirname(save_path)
        if save_folder != '':
            os.makedirs(save_folder, exist_ok=True)

        cv2.imwrite(save_path, img)
        print(f"Saved image to: {save_path}")



if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = "weights/yolov8s.pt"
    input_image = "upscaled_coastguard.png"
    output_image_path = "yolo_coastguard.png"  
    run_yolo_person_detection(model_path, input_image, output_image_path)
    print("Detection complete.")
