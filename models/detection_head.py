# models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Detection Head
# -----------------------------
class DetectionHead(nn.Module):
    """
    YOLO-style detection head.
    Output format: [B, A, H, W, 5 + num_classes + 1(orientation)]
    5 -> [dx,dy,dw,dh,obj_conf,class_probs...], plus orientation scalar (regression)
    """
    def __init__(self, in_channels=512, num_classes=20, anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_outputs = 5 + num_classes + 1  # 4 bbox + obj + classes + orientation
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.SiLU()
        )
        self.pred = nn.Conv2d(in_channels//2, self.anchors * self.num_outputs, 1)

    def forward(self, feat):
        x = self.conv_reduce(feat)
        pred = self.pred(x)
        B, C, H, W = pred.shape
        pred = pred.view(B, self.anchors, self.num_outputs, H, W)
        pred = pred.permute(0,1,3,4,2).contiguous()  # [B, A, H, W, num_outputs]
        return pred


# -----------------------------
# MAE Decoder (basic conv decoder)
# -----------------------------
class MAEDecoder(nn.Module):
    def __init__(self, token_dim=512, out_channels=512, patch_size=1, num_upsample=4):
        super().__init__()
        # Map tokens back to patch feature channels
        self.linear = nn.Linear(token_dim, out_channels)
        # simple upsampling decoder (conv transpose or conv blocks)
        layers = []
        for _ in range(num_upsample):
            layers.append(nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels//2))
            layers.append(nn.SiLU())
            out_channels = out_channels//2
        self.decoder = nn.Sequential(*layers)
        self.final = nn.Conv2d(out_channels, 3, kernel_size=1)  # reconstruct RGB patch (coarse)

    def forward(self, tokens, H, W):
        """
        tokens: [B, N, C]  (all tokens or masked-encoded tokens)
        H, W: spatial dims
        """
        B, N, C = tokens.shape
        x = self.linear(tokens)  # [B, N, out_ch]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, out_ch, H, W]
        x = self.decoder(x)
        out = self.final(x)
        return out  # [B, 3, H_up, W_up] approximate reconstruction


# -----------------------------
# Projection heads (DINO / CLIP)
# -----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, proj_dim=256, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        # x: [B, C] or [B, N, C] (we'll pool over tokens when needed)
        if x.dim() == 3:
            x = x.mean(1)  # pool tokens -> [B, C]
        out = self.mlp(x)
        out = F.normalize(out, dim=-1)
        return out

