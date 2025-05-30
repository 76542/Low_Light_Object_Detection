# Low-Light Object Detection
Enhance, super-resolve, and detect objects in challenging low-light scenes with a modular PyTorch pipeline.
Uses Zero-DCE for illumination correction, ESRGAN/SRGAN for upscaling, and YOLOv8 for robust object detection.

# Pipeline Overview
Low-Light Image → **Zero-DCE** (Enhance) → **SRGAN/ESRGAN** (Super-Resolve) → **YOLOv8** (Detect Objects) → **Final Output**

# Features
Zero-DCE: State-of-the-art deep curve estimation for unsupervised low-light enhancement (Zero-DCE Paper).

SRGAN/ESRGAN: GAN-based upscaling for ultra-sharp, high-resolution restoration.

YOLOv8: Real-time object detection, fully compatible with Apple Silicon (M1/M2) or CUDA GPUs.

Easy-to-Run Scripts: Modular code for each stage. Plug in your own images, weights, or models.

Runs on Mac (M1/M2), Windows, Linux.

# Folder Structure 


# QuickStart
1. Clone and Set Up
bash
Copy
Edit
git clone https://github.com/<yourusername>/Low_Light_Object_Detection.git
cd Low_Light_Object_Detection
python3.9 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

2. Download/Prepare Weights
zerodce_epoch_50.pth — Train with LOL dataset using train_zerodce.py.

RRDB_ESRGAN_x4.pth — Download ESRGAN weights here.

yolov8n.pt — Download YOLOv8n here.

Place all weights in the weights/ folder.

3. Enhance an Image (Zero-DCE)
bash
Copy
Edit
python pipeline/zerodce.py
Input: data/test/input/your_image.jpeg

Output: data/test/enhanced/your_image.jpeg

4. Super-Resolution (ESRGAN/ SRGAN)
bash
Copy
Edit
python pipeline/srgan.py
Input: data/test/enhanced/your_image.jpeg
Output: data/test/srgan/your_image.jpeg

5. Object Detection (YOLOv8)
bash
Copy
Edit
python pipeline/yolo.py
Input: data/test/srgan/your_image.jpeg (or use enhanced output directly)
Output: data/test/yolo/your_image.jpeg (with bounding boxes)

# Training Zero-DCE (Optional)
To train Zero-DCE on the LOL dataset:

bash
Copy
Edit
python train_zerodce.py
Saves model checkpoints to weights/

# Customization
SRGAN/ESRGAN: Swap with any generator weights. Use your own images for super-resolution.

YOLOv8: Use custom-trained weights for your object or class of interest.

Pipeline: Chain all scripts or call from a single master script for full automation.

# Tips & Notes
Apple Silicon (M1/M2) fully supported (via torch.device('mps')).

Adjust input/output paths for your needs in each script.

Check results in each subfolder—every stage is modular!

# References
Zero-DCE: Zero-Reference Deep Curve Estimation for Low-Light Enhancement (CVPR 2020)

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (ECCV Workshops 2018)

YOLOv8 by Ultralytics
