import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Set this to your full dataset path

def get_image_label_dataframe(image_dir):
    image_paths = []
    labels = []

    for root, _, files in os.walk(image_dir):
        for filename in sorted(files):
            if filename.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)
                label = 1 if "fake" in root.lower() or "fake" in filename.lower() else 0
                labels.append(label)

    return pd.DataFrame({'image': image_paths, 'label': labels})

class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label.unsqueeze(0)


train_transform = A.Compose([
    A.Resize(224,224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def data_pipeline(full_path, train_transform, val_transform, batch=32, val_split=0.2, seed=42,val_already_split=False,val_path=None,train_path=None):

    if not val_already_split:
        df = get_image_label_dataframe(full_path)
        train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['label'], random_state=seed)   
    else:
        train_df = get_image_label_dataframe(train_path)
        val_df = get_image_label_dataframe(val_path)

    train_dataset = ClassificationDataset(train_df["image"].tolist(), train_df["label"].tolist(), train_transform)
    val_dataset = ClassificationDataset(val_df["image"].tolist(), val_df["label"].tolist(), val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

if __name__== "__main__":
    full_dataset_path = "/home/manik/Documents/datasets/images"

    train_loader, val_loader = data_pipeline(full_dataset_path, train_transform, val_transform, batch=64)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")