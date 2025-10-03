# utils/detection_loss.py
import torch
import torch.nn.functional as F

def bbox_xywh_to_xyxy(box):
    # box: [..., 4] x,y,w,h (cx,cy,w,h) normalized to image dims or absolute
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return torch.stack([x1,y1,x2,y2], dim=-1)

def compute_iou(boxes1, boxes2, eps=1e-7):
    # boxes1: [N,4] x1y1x2y2 ; boxes2: [M,4]
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    # expand
    boxes1 = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2 = boxes2.unsqueeze(0).expand(N, M, 4)
    x1 = torch.max(boxes1[...,0], boxes2[...,0])
    y1 = torch.max(boxes1[...,1], boxes2[...,1])
    x2 = torch.min(boxes1[...,2], boxes2[...,2])
    y2 = torch.min(boxes1[...,3], boxes2[...,3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (boxes1[...,2]-boxes1[...,0]).clamp(min=0) * (boxes1[...,3]-boxes1[...,1]).clamp(min=0)
    area2 = (boxes2[...,2]-boxes2[...,0]).clamp(min=0) * (boxes2[...,3]-boxes2[...,1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    return inter / union

def ciou_loss(pred_boxes_xyxy, target_boxes_xyxy, eps=1e-7):
    """
    Calculate 1 - CIoU between two boxes
    Inputs are [N,4] x1y1x2y2
    Returns mean CIoU loss
    """
    # IoU
    iou = compute_iou(pred_boxes_xyxy, target_boxes_xyxy)
    # For pairing purpose: keep elementwise for matched pairs only
    # Compute centers and sizes
    px = (pred_boxes_xyxy[...,0] + pred_boxes_xyxy[...,2]) / 2
    py = (pred_boxes_xyxy[...,1] + pred_boxes_xyxy[...,3]) / 2
    pw = (pred_boxes_xyxy[...,2] - pred_boxes_xyxy[...,0]).clamp(min=eps)
    ph = (pred_boxes_xyxy[...,3] - pred_boxes_xyxy[...,1]).clamp(min=eps)

    tx = (target_boxes_xyxy[...,0] + target_boxes_xyxy[...,2]) / 2
    ty = (target_boxes_xyxy[...,1] + target_boxes_xyxy[...,3]) / 2
    tw = (target_boxes_xyxy[...,2] - target_boxes_xyxy[...,0]).clamp(min=eps)
    th = (target_boxes_xyxy[...,3] - target_boxes_xyxy[...,1]).clamp(min=eps)

    # center distance
    rho2 = (px - tx)**2 + (py - ty)**2
    # enclosing box
    c_x1 = torch.min(pred_boxes_xyxy[...,0], target_boxes_xyxy[...,0])
    c_y1 = torch.min(pred_boxes_xyxy[...,1], target_boxes_xyxy[...,1])
    c_x2 = torch.max(pred_boxes_xyxy[...,2], target_boxes_xyxy[...,2])
    c_y2 = torch.max(pred_boxes_xyxy[...,3], target_boxes_xyxy[...,3])
    c_w = (c_x2 - c_x1).clamp(min=eps)
    c_h = (c_y2 - c_y1).clamp(min=eps)
    c2 = c_w**2 + c_h**2

    # aspect ratio term
    v = (4 / (3.141592653589793**2)) * (torch.atan(tw / th) - torch.atan(pw / ph))**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / (c2 + eps)) - alpha * v
    loss = 1 - ciou
    return loss.mean()

def match_anchors_to_targets(anchor_boxes_xyxy, target_boxes_xyxy, iou_threshold=0.5):
    """
    Simple greedy matching:
    - anchor_boxes: [A,4]
    - target_boxes: [T,4]
    returns matched pairs lists
    """
    A = anchor_boxes_xyxy.size(0)
    T = target_boxes_xyxy.size(0)
    if T == 0 or A == 0:
        return [], []

    ious = compute_iou(anchor_boxes_xyxy, target_boxes_xyxy)  # [A,T]
    ious_flat = ious.view(-1)
    # sort descending
    vals, idxs = torch.sort(ious_flat, descending=True)
    matched_anchors = set()
    matched_targets = set()
    matches = []
    for v, idx in zip(vals.cpu().numpy(), idxs.cpu().numpy()):
        if v < iou_threshold:
            break
        a = idx // T
        t = idx % T
        if a in matched_anchors or t in matched_targets:
            continue
        matched_anchors.add(a)
        matched_targets.add(t)
        matches.append((a, t))
        if len(matched_targets) == T:
            break
    return matches  # list of (anchor_idx, target_idx)

