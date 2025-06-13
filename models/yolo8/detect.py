import torch
from ultralytics import YOLO

def yolo_detect(
    weights_path,
    image_path,
    output_dir="runs/detect",
    device=None,
    conf=0.25
):
    """
    General YOLOv8 detection module.
    Args:
        weights_path (str): Path to YOLOv8 .pt weights (e.g., yolov8s.pt)
        image_path (str): Path to image or folder
        output_dir (str): Output directory
        device (str): 'cuda', 'mps', 'cpu', or None (auto)
        conf (float): Confidence threshold
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(weights_path)
    results = model(
        image_path,
        conf=conf,
        save=True,
        project=output_dir,
        device=device
    )

    for r in results:
        print(r)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/yolo8/yolov8s.pt")
    parser.add_argument("--image", type=str, required=True, help="Image or folder to run detection on")
    parser.add_argument("--outdir", type=str, default="runs/detect", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    yolo_detect(
        args.weights,
        args.image,
        output_dir=args.outdir,
        conf=args.conf
    )
