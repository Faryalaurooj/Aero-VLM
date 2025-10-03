import torch
import cv2
import yaml
import argparse
from models.aerovlm import AeroVLM
from data.transforms import preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/default.yaml")
parser.add_argument("--weights", type=str, required=True)
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AeroVLM(num_classes=cfg["num_classes"]).to(device)
model.load_state_dict(torch.load(args.weights))
model.eval()

# Image preprocess
img, tensor = preprocess(args.image, cfg["img_size"])
tensor = tensor.unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    preds = model(tensor)

print("Predictions:", preds)
cv2.imshow("Aircraft", img)
cv2.waitKey(0)

