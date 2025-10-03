import torch
import yaml
import argparse
from torch.utils.data import DataLoader
from models.aerovlm import AeroVLM
from data.dataset import AircraftDataset
from utils.engine import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/default.yaml")
parser.add_argument("--weights", type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dataset = AircraftDataset(cfg["val_data"], cfg["img_size"])
val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

model = AeroVLM(num_classes=cfg["num_classes"]).to(device)
model.load_state_dict(torch.load(args.weights))
model.eval()

evaluate(model, val_loader, device)

