import cv2
import torch
import torchvision.transforms as T

def preprocess(path, img_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    tensor = transform(img)
    return img, tensor

