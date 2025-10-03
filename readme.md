# 🛩️ Aero-VLM: Hybrid Vision–Language Model for Fine-Grained Aircraft Recognition in Remote Sensing

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Abstract
Fine-grained aircraft recognition in high-resolution remote sensing imagery is critical for defense, surveillance, and monitoring, yet remains highly challenging due to **intra-class similarity, small object size, occlusion, camouflage, and cluttered backgrounds**.  

We present **Aero-VLM**, a hybrid deep learning framework that integrates:
- CNN feature extraction  
- Transformer encoders  
- Multiscale deformable attention  
- Cascaded fusion + detection refinement  
- CLIP-style vision–language supervision  

Alongside the model, we introduce **Aero-RSI**, a curated high-resolution benchmark dataset with 6,000+ images across 10 aircraft categories. Aero-VLM achieves **93.7% mAP** and **93.2% F1-score**, surpassing YOLOv11 (+3.6% mAP, +2.6% F1) while reducing inference latency by 23.1%.  

---

## 📂 Project Structure
aero-vlm/
│── data/ # Datasets (Aero-RSI, FGSC-23, etc.)
│── models/
│ ├── backbone.py # CNN + Transformer hybrid backbone
│ ├── transformer.py # Vision transformer encoder
│ ├── fusion.py # Multiscale + cascaded fusion
│ ├── detection_head.py # Detection & classification head
│ └── aero_vlm.py # Full model assembly
│── utils/
│ ├── dataset.py # Custom dataset loader
│ ├── transforms.py # Data augmentations
│ ├── metrics.py # mAP, F1-score, PR curves
│── train.py # Training script
│── evaluate.py # Evaluation script
│── inference.py # Run inference on new images
│── export.py # Export ONNX / TorchScript models
│── requirements.txt
│── README.md



---

##  Installation
```bash
git clone https://github.com/yourusername/aero-vlm.git
cd aero-vlm

# Create environment
conda create -n aero-vlm python=3.9 -y
conda activate aero-vlm

# Install dependencies
pip install -r requirements.txt

##  Dataset Setup
# Aero-RSI (ours)

Organize as:
data/Aero-RSI/
│── images/
│   ├── train/
│   ├── val/
│   ├── test/
│── annotations/
│   ├── train.json
│   ├── val.json
│   ├── test.json

Also supports:

FGSC-23

##  Training

python train.py \
  --data data/Aero-RSI/annotations/train.json \
  --img-dir data/Aero-RSI/images/train \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4 \
  --save-dir runs/train/aero-vlm
  
  Resume training:
  
  
python train.py --resume runs/train/aero-vlm/last_checkpoint.pth

## Evaluation

python evaluate.py \
  --weights runs/train/aero-vlm/best_model.pth \
  --data data/Aero-RSI/annotations/test.json \
  --img-dir data/Aero-RSI/images/test

Metrics computed:

mAP@0.5

AP_small

F1-score

PR curves

## Inference

python inference.py \
  --weights runs/train/aero-vlm/best_model.pth \
  --img data/sample_aircraft.jpg \
  --output results/


## Edge Deployment

# Export ONNX
python export.py --weights best_model.pth --format onnx

# Export TorchScript
python export.py --weights best_model.pth --format torchscript

## Results
| Model               | mAP@0.5  | AP_small | F1       | Latency (ms) |
| ------------------- | -------- | -------- | -------- | ------------ |
| CLIP                | 58.2     | 42.1     | 60.4     | 120          |
| GLIP                | 72.3     | 54.7     | 70.6     | 95           |
| Kosmos-2            | 75.8     | 58.3     | 73.9     | 110          |
| VisionLLM           | 78.1     | 61.0     | 76.2     | 130          |
| **Aero-VLM (ours)** | **84.6** | **71.2** | **83.1** | 100          |


| Model               | FGSC-23 mAP@0.5 | DOTA (aircraft) mAP@0.5 |
| ------------------- | --------------- | ----------------------- |
| GLIP                | 69.4            | 63.8                    |
| Kosmos-2            | 72.5            | 66.2                    |
| VisionLLM           | 74.1            | 68.0                    |
| **Aero-VLM (ours)** | **80.3**        | **76.9**                |

| Shots per class | 0    | 1    | 5    | 10   |
| --------------- | ---- | ---- | ---- | ---- |
| mAP@0.5         | 84.6 | 88.9 | 91.8 | 92.9 |
| AP_small        | 71.2 | 76.5 | 80.4 | 81.9 |
| F1              | 83.1 | 86.7 | 89.6 | 90.3 |










