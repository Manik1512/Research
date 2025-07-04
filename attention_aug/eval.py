from model import my_model
from utils.metrics import compute_metrics
from utils.data_pipeline import get_image_label_dataframe,val_transform, ClassificationDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import parameters


def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=my_model(attention_heads=parameters.ATTENTION_HEADS,batch_size=parameters.BATCH_SIZE,threshold=0.5,lambda_reg=parameters.REG_LAMDA)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model



def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    maps = []
    augmented_maps = []


    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()


            logits,augmented_logits, attention_maps, augmented_attention_maps,pooled_features = model.predict(inputs)
            

            P_raw = torch.sigmoid(logits)
            P_fine_grained = torch.sigmoid(augmented_logits)
            outputs = (P_raw + P_fine_grained) / 2.0

            y_true.append(labels.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())    

            maps.append(attention_maps.cpu().numpy())
            augmented_maps.append(augmented_attention_maps.cpu().numpy())

    return compute_metrics(y_true, y_pred) ,maps, augmented_maps


if __name__ == '__main__':

    test_path="/home/manik/Documents/datasets/Dataset/Test"
    # test_path="/home/manik/Documents/datasets/images"

    model=load_model(checkpoint_path="/home/manik/Documents/model_results/checkpoints/checkpoint_epoch_5.pth")

    test_df = get_image_label_dataframe(test_path)
    test_dataset = ClassificationDataset(test_df["image"].tolist(), test_df["label"].tolist(), val_transform)
    test_loader = DataLoader(test_dataset, batch_size=parameters.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics,maps,augmented_maps = evaluate(model, test_loader, device)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
