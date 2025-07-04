import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import numpy as np 



path="/home/manik/Documents/model_results"

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

import os 
def log_metrics(epoch, metrics, train_loss, val_loss, log_file=os.path.join(path, "metrics_log_freezed.csv")):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["epoch", "accuracy", "precision", "recall", "f1", "train_loss", "val_loss"])
        writer.writerow([epoch] + list(metrics.values()) + [train_loss, val_loss])

def compute_metrics(y_true, y_pred):
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_pred_label = (y_pred > 0.5).astype(int)

    return {
        'Accuracy': accuracy_score(y_true, y_pred_label),
        'Precision': precision_score(y_true, y_pred_label),
        'Recall': recall_score(y_true, y_pred_label),
        'F1 Score': f1_score(y_true, y_pred_label)
    }

def save_checkpoint(model, optimizer, epoch, checkpoint_dir=os.path.join(path, "checkpoints_freezed")):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")



