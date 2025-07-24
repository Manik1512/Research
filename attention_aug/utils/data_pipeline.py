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
import pytorch_lightning as pl


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


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self,
                 full_path=None,
                 train_transform=None,
                 val_transform=None,
                 batch_size=32,
                 val_split=0.2,
                 seed=42,
                 val_already_split=False,
                 val_path=None,
                 train_path=None,
                 num_workers=4):
        super().__init__()
        self.full_path = full_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.val_already_split = val_already_split
        self.train_path = train_path
        self.val_path = val_path
        self.num_workers = num_workers

    def setup(self, stage=None):
        if not self.val_already_split:
            df = get_image_label_dataframe(self.full_path)
            train_df, val_df = train_test_split(df, test_size=self.val_split,
                                                stratify=df['label'], random_state=self.seed)
        else:
            train_df = get_image_label_dataframe(self.train_path)
            val_df = get_image_label_dataframe(self.val_path)

        self.train_dataset = ClassificationDataset(
            train_df["image"].tolist(),
            train_df["label"].tolist(),
            self.train_transform
        )

        self.val_dataset = ClassificationDataset(
            val_df["image"].tolist(),
            val_df["label"].tolist(),
            self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )



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


def data_pipeline_pl(full_path, train_transform, val_transform, batch=32, val_split=0.2, seed=42,val_already_split=False,val_path=None,train_path=None):
    if not val_already_split:
        df = get_image_label_dataframe(full_path)
        train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['label'], random_state=seed)   
    else:
        train_df = get_image_label_dataframe(train_path)
        val_df = get_image_label_dataframe(val_path)

    
if __name__== "__main__":
    full_dataset_path = "/home/manik/Documents/datasets/images"

    datamodule = ClassificationDataModule(
        full_path=full_dataset_path,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=64
    )

    datamodule.setup()  # Optional to preload for checks

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")