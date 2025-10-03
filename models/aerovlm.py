import torch.nn as nn
from models.backbone import HybridBackbone
from models.detection_head import DetectionHead

class AeroVLM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = HybridBackbone(embed_dim=512, depth=6, heads=8)
        self.head = DetectionHead(in_channels=512, num_classes=num_classes, anchors=3)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out

