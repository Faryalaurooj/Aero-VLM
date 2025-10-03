# data/dataset.py
#A dataset that:

#reads COCO-like annotation JSON (adapt as needed),

#returns:

#image tensor,

#bboxes tensor (N,4) in xywh normalized,

#class_ids tensor (N,),

#orientation tensor (N,) in radians normalized to [-pi,pi],

#text_prompts list (strings) — simple prompts using class names,

#view1, view2 (two augmentations for DINO),

#masked_input if you want.

#This uses torchvision transforms (you can swap to albumentations).
import os
import json
import random
import math
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class AircraftDataset(torch.utils.data.Dataset):
    """
    Expects a COCO-style JSON with:
       images: [{id, file_name, width, height}, ...]
       annotations: [{image_id, bbox [x,y,w,h], category_id, orientation (optional)}, ...]
       categories: [{id, name}, ...]
    """
    def __init__(self, root_dir, ann_file, img_size=640, classes=None, dino_augment=True):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        self.images = {img['id']: img for img in coco['images']}
        self.anns_by_image = {}
        for ann in coco['annotations']:
            self.anns_by_image.setdefault(ann['image_id'], []).append(ann)
        self.image_ids = list(self.images.keys())
        # class id -> name
        self.cat = {c['id']: c['name'] for c in coco['categories']}
        self.classes = classes if classes else list(self.cat.values())
        self.dino_augment = dino_augment

        # transforms
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

        # DINO augmentations: simple random crops + color jitter (replace with stronger pipeline if you wish)
        self.aug1 = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4,1.0)),
            T.ColorJitter(0.4,0.4,0.4,0.1),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        self.aug2 = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2,1.0)),
            T.ColorJitter(0.8,0.8,0.8,0.2),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, fname):
        img = Image.open(fname).convert('RGB')
        return img

    def _process_annotations(self, anns, orig_w, orig_h):
        boxes = []
        labels = []
        orients = []
        for ann in anns:
            x,y,w,h = ann['bbox']
            # normalize to [0,1] relative to original image
            boxes.append([ (x + w/2) / orig_w, (y + h/2) / orig_h, w / orig_w, h / orig_h ])  # cx,cy,w,h
            labels.append(ann['category_id'])
            orients.append(ann.get('orientation', 0.0))  # assume radians or degrees; user adjust
        if len(boxes) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            orients = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            orients = torch.tensor(orients, dtype=torch.float32)
        return boxes, labels, orients

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        meta = self.images[img_id]
        img_path = os.path.join(self.root_dir, meta['file_name'])
        img = self._load_image(img_path)
        anns = self.anns_by_image.get(img_id, [])
        boxes, labels, orients = self._process_annotations(anns, meta['width'], meta['height'])

        # base tensor (for detection / MAE)
        img_det = self.base_transform(img)  # [3,H,W]
        # two augmented views for DINO
        view1 = self.aug1(img) if self.dino_augment else img_det
        view2 = self.aug2(img) if self.dino_augment else img_det

        # text prompts: simple "a photo of a {class_name}"
        text_prompts = []
        for lbl in labels.tolist():
            cname = self.cat.get(int(lbl), "aircraft")
            text_prompts.append(f"aerial image of a {cname}")
        if len(text_prompts) == 0:
            text_prompts = ["aerial image of an aircraft"]

        sample = {
            "image": img_det,
            "view1": view1,
            "view2": view2,
            "bboxes": boxes,
            "labels": labels,
            "orientation": orients,
            "text_prompts": text_prompts
        }
        return sample

