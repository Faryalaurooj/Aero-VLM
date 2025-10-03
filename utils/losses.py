# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Simple detection losses (stubs)
# -----------------------------
def bbox_loss(pred_boxes, target_boxes):
    """
    pred_boxes: [B, A, H, W, 4] (dx,dy,dw,dh)
    target_boxes: same shape
    Very simple L1 loss here — replace with CIoU/GIoU for production.
    """
    return F.l1_loss(pred_boxes, target_boxes, reduction='mean')

def obj_loss(pred_obj, target_obj):
    return F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='mean')

def cls_loss(pred_cls_logits, target_cls):
    # pred_cls_logits: [B, A, H, W, num_classes]
    # target_cls: index or one-hot
    # flatten and compute CE
    B, A, H, W, C = pred_cls_logits.shape
    pred = pred_cls_logits.view(-1, C)
    tgt = target_cls.view(-1).long()
    return F.cross_entropy(pred, tgt, ignore_index=-1)

def orientation_loss(pred_ori, target_ori):
    # regression MSE for orientation angle (in radians or normalized)
    return F.mse_loss(pred_ori, target_ori, reduction='mean')


# -----------------------------
# MAE reconstruction loss
# -----------------------------
def mae_loss(recon, target):
    # recon, target: [B, 3, H, W]
    return F.mse_loss(recon, target, reduction='mean')


# -----------------------------
# CLIP-style alignment loss (InfoNCE / cosine)
# -----------------------------
def clip_alignment_loss(img_emb, text_emb, temperature=0.07):
    # img_emb, text_emb: [B, D] normalized
    logits = img_emb @ text_emb.t() / temperature
    labels = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2.0


# -----------------------------
# DINO-style contrastive loss (placeholder)
# -----------------------------
def dino_loss(student_proj, teacher_proj, center=None, temp_student=0.1, temp_teacher=0.04):
    """
    This is a simplified placeholder. Full DINO uses centering + teacher momentum + sharpened teacher
    """
    # student_proj, teacher_proj: [B, D] normalized
    logits = student_proj @ teacher_proj.t() / temp_student
    labels = torch.arange(student_proj.size(0), device=student_proj.device)
    loss = F.cross_entropy(logits, labels)
    return loss

