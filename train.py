# train.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from models.backbone import HybridBackbone
from models.heads import DetectionHead, MAEDecoder, ProjectionHead
from utils.losses import (bbox_loss, obj_loss, cls_loss, orientation_loss, 
                          mae_loss, clip_alignment_loss, dino_loss)

# Placeholder dataset import (you'll implement dataset to return images, bboxes, classes, text prompts)
from data.dataset import AircraftDataset

def train_one_epoch(model_parts, loader, optimizers, device, epoch, cfg):
    backbone, det_head, mae_decoder, proj_head = model_parts
    optimizer = optimizers

    backbone.train(); det_head.train(); mae_decoder.train(); proj_head.train()

    for batch in loader:
        # Expected batch structure (you must implement in dataset):
        # images: [B,3,H,W], bboxes, class_ids, orientation, text_prompts
        images, bboxes, class_ids, orientation, text_prompts = batch
        images = images.to(device)

        # -------------------
        # 1. Forward: get full features and masked tokens (for MAE)
        # -------------------
        mask_ratio = cfg.get("mae_mask_ratio", 0.6)
        full_feat, enc_vis, mask, (h,w), ids_restore = backbone(images, mask_ratio=mask_ratio)

        # Detection predictions (on full_feat)
        preds = det_head(full_feat)  # [B, A, Hf, Wf, outputs]
        # (You must decode preds and prepare target tensors compatible with losses)
        # Prepare dummy targets for illustration (replace with proper target tensors)
        # bbox_pred = preds[..., :4]; obj_pred = preds[...,4]; cls_logits = preds[...,5:5+num_classes]; ori_pred = preds[..., -1]

        # -------------------
        # 2. MAE reconstruction path (reconstruct masked patches)
        # -------------------
        if enc_vis is not None:
            # enc_vis: [B, N_vis, C] -> we rely on ids_restore to reconstruct full set in decoder
            # For simplicity, run decoder on enc_all tokens by running backbone again without masking
            enc_all_feat, _, _, (h,w), _ = backbone(images, mask_ratio=0.0)
            tokens, _ = backbone.patchify(enc_all_feat) if hasattr(backbone, 'patchify') else (None, (h,w))
            # decode: (here we decode tokens to image-features to compute L_MAE against input)
            recon = mae_decoder(tokens, h, w)  # [B,3,H_up,W_up]
            L_MAE = mae_loss(recon, images)    # coarse recon loss
        else:
            L_MAE = torch.tensor(0.0, device=device)

        # -------------------
        # 3. DINO / projection losses
        # -------------------
        # get pooled image embedding
        # For simplicity compute global pooled embedding from full_feat
        img_tokens, _ = backbone.patchify(backbone.proj(backbone.cnn(images)))
        # run transformer on all tokens to get features for proj head
        enc_all = backbone.transformer(img_tokens)
        img_proj = proj_head(enc_all)  # [B, D] (pooled inside proj if needed)
        # Dummy teacher proj (in real DINO you maintain a momentum teacher)
        teacher_proj = img_proj.detach()

        L_DINO = dino_loss(img_proj, teacher_proj)

        # -------------------
        # 4. CLIP-style text alignment
        # -------------------
        # You must provide text prompt embeddings through a tokenizer + text encoder (not included here)
        # For placeholder, create random text embeddings:
        text_emb = torch.randn_like(img_proj).to(device)
        L_ALIGN = clip_alignment_loss(img_proj, text_emb)

        # -------------------
        # 5. Detection losses (placeholders)
        # -------------------
        # Placeholder zeros - you must decode preds and compare with targets
        L_bbox = torch.tensor(0.0, device=device)
        L_obj = torch.tensor(0.0, device=device)
        L_cls = torch.tensor(0.0, device=device)
        L_ori = torch.tensor(0.0, device=device)

        # -------------------
        # Total loss
        # -------------------
        lambda_mae = cfg.get("lambda_mae", 1.0)
        lambda_dino = cfg.get("lambda_dino", 1.0)
        lambda_align = cfg.get("lambda_align", 1.0)
        lambda_det = cfg.get("lambda_det", 1.0)

        total_loss = lambda_det*(L_bbox + L_obj + L_cls + L_ori) + lambda_mae * L_MAE + lambda_dino * L_DINO + lambda_align * L_ALIGN

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done. Loss: {total_loss.item():.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "mae_mask_ratio": 0.6,
        "lambda_mae": 1.0,
        "lambda_dino": 1.0,
        "lambda_align": 1.0,
        "lambda_det": 1.0
    }

    # instantiate modules
    backbone = HybridBackbone(embed_dim=512, depth=8, heads=8).to(device)
    det_head = DetectionHead(in_channels=512, num_classes=20).to(device)
    mae_decoder = MAEDecoder(token_dim=512, out_channels=512).to(device)
    proj_head = ProjectionHead(in_dim=512, proj_dim=256).to(device)

    model_parts = (backbone, det_head, mae_decoder, proj_head)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(det_head.parameters()) + list(mae_decoder.parameters()) + list(proj_head.parameters()),
        lr=1e-4
    )

    # dataset placeholder
    train_dataset = AircraftDataset("data/train", img_size=640)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    for epoch in range(10):
        train_one_epoch(model_parts, train_loader, optimizer, device, epoch, cfg)

if __name__ == "__main__":
    main()

