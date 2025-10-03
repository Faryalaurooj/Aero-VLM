# models/backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# -----------------------------
# CNN Stem
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CNNStem(nn.Module):
    """Multi-stage CNN stem producing a single high-level feature map"""
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        self.layer1 = ConvBlock(3, channels[0], 3, 2, 1)   # H/2
        self.layer2 = ConvBlock(channels[0], channels[1], 3, 2, 1)  # H/4
        self.layer3 = ConvBlock(channels[1], channels[2], 3, 2, 1)  # H/8
        self.layer4 = ConvBlock(channels[2], channels[3], 3, 2, 1)  # H/16

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, C, H/16, W/16]


# -----------------------------
# Transformer blocks
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C]
        h = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x = h + self.drop(x_attn)

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h + self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim=512, depth=8, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads=heads) for _ in range(depth)])
        self.dim = dim

    def forward(self, x):
        # x: [B, N, C]
        for blk in self.layers:
            x = blk(x)
        return x  # [B, N, C]


# -----------------------------
# Hybrid Backbone
# -----------------------------
class HybridBackbone(nn.Module):
    """
    Returns:
      full_feat: [B, C, H, W]  (refined)
      mae_tokens: (masked tokens) [B, M, C] or None
      token_meta: (H, W) to reproject tokens
    """
    def __init__(self, embed_dim=512, depth=8, heads=8, channels=512):
        super().__init__()
        self.cnn = CNNStem(channels=[64,128,256,channels])
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=1)  # project to transformer dim
        self.transformer = TransformerEncoder(dim=embed_dim, depth=depth, heads=heads)
        self.embed_dim = embed_dim

    def patchify(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int]]:
        # feat: [B, C, H, W] -> tokens [B, H*W, C]
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)  # [B, N, C]
        return tokens, (H, W)

    def unpatchify(self, tokens: torch.Tensor, H:int, W:int) -> torch.Tensor:
        # tokens: [B, N, C] -> [B, C, H, W]
        B, N, C = tokens.shape
        assert N == H * W
        x = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return x

    def random_masking(self, tokens: torch.Tensor, mask_ratio: float):
        """
        tokens: [B, N, C]; returns visible_tokens, mask, ids_restore, ids_keep
        """
        B, N, C = tokens.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=tokens.device)  # noise in [0,1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        visible_tokens = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        mask = torch.ones([B, N], device=tokens.device)
        mask[:, :len_keep] = 0
        # unshuffle mask to original ordering
        mask = torch.gather(mask, 1, ids_restore)
        return visible_tokens, mask, ids_restore, ids_keep

    def forward(self, x: torch.Tensor, mask_ratio: Optional[float]=0.0):
        """
        x: [B, 3, H, W]
        mask_ratio: for MAE-style masking (0.0 => no masking)
        """
        feat = self.cnn(x)          # [B, C, h, w]
        proj = self.proj(feat)      # [B, embed_dim, h, w]
        tokens, (h, w) = self.patchify(proj)  # [B, N, C]

        if mask_ratio and mask_ratio > 0.0:
            visible_tokens, mask, ids_restore, ids_keep = self.random_masking(tokens, mask_ratio)
            # transformer on visible tokens
            enc_vis = self.transformer(visible_tokens)  # [B, N_vis, C]
            # reconstruct full token set by placing encoded visible tokens back (decoder head elsewhere)
            # also produce a version of full tokens by running transformer on all tokens (for DINO)
            enc_all = self.transformer(tokens)
            # project back to spatial map only for enc_all
            full_feat = self.unpatchify(enc_all, h, w)  # [B, C, h, w]
            return full_feat, enc_vis, mask, (h, w), ids_restore
        else:
            enc_all = self.transformer(tokens)  # [B, N, C]
            full_feat = self.unpatchify(enc_all, h, w)
            return full_feat, None, None, (h, w), None

