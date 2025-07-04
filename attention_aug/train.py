import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.data_pipeline import data_pipeline,train_transform,val_transform
from utils.metrics import EarlyStopping, log_metrics, compute_metrics, save_checkpoint
from utils.losses import AttentionRegularizationLoss
from torch import nn
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import my_model
import os 
import parameters
from utils.augment import attention_augment


# batch_size=16
attention= attention_augment(batch_size=parameters.BATCH_SIZE, attention_heads=parameters.ATTENTION_HEADS,threshld=0.5)






# reg_loss=AttentionRegularizationLoss(parameters.ATTENTION_HEADS,1280)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def compute_total_loss(logits, augmented_logits, pooled_features, labels,lambda_reg,model):
    logits = torch.sigmoid(logits)
    augmented_logits = torch.sigmoid(augmented_logits)
    # loss_class = F.binary_cross_entropy(logits, labels) + F.binary_cross_entropy(augmented_logits, labels)
    # loss_reg = model.regul_loss(pooled_features)
    # return loss_class + lambda_reg * loss_reg
    raw_loss = F.binary_cross_entropy(logits, labels)
    fine_loss = F.binary_cross_entropy(augmented_logits, labels)
    classification_loss = raw_loss + fine_loss
    regularization_loss = model.regul_loss(pooled_features)
    total_loss = classification_loss + lambda_reg * regularization_loss
    
    return classification_loss , total_loss



def train_model(model, train_loader, val_loader, optimizer, num_epochs=50, patience=5,reg_lamda=0.05):
    early_stopping = EarlyStopping(patience=patience)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        class_train_loss=0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits,augmented_logits,  pooled_features = model(imgs)

            class_loss,loss = compute_total_loss(logits, augmented_logits, pooled_features, labels,lambda_reg=reg_lamda,model=model)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            class_train_loss += class_loss.item()
            train_pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)
             
                logits,augmented_logits,pooled_features= model(imgs)
                P_raw= torch.sigmoid(logits)
                P_fine_grained = torch.sigmoid(augmented_logits)

                outputs =(P_raw + P_fine_grained) / 2.0

                val_class_loss,val_loss_batch = compute_total_loss(logits,augmented_logits, pooled_features, labels,lambda_reg=reg_lamda,model=model)
                val_loss += val_class_loss.item()
                y_true.append(labels.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())
                val_pbar.set_postfix(loss=val_class_loss.item())


        avg_train_class_loss=class_train_loss / len(train_loader)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(y_true, y_pred)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train_cls Loss: {avg_train_class_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"val metrics => {metrics}")

        log_metrics(epoch, metrics, avg_train_loss, avg_val_loss)
        save_checkpoint(model, optimizer, epoch)
        scheduler.step(avg_val_loss)

        if early_stopping(avg_val_loss):
            print("Early stopping triggered.")
            break



if __name__ == "__main__":
    train_loader,val_loader = data_pipeline(full_path=None,
                                            train_transform=train_transform,
                                            val_transform=val_transform,
                                            batch=parameters.BATCH_SIZE,
                                            val_already_split=True,
                                            train_path=os.path.join(parameters.DATASET_PATH, "Train"),
                                            val_path=os.path.join(parameters.DATASET_PATH, "Validation")
                                            )



    model=my_model(attention_heads=parameters.ATTENTION_HEADS,batch_size=parameters.BATCH_SIZE,threshold=0.5,lambda_reg=parameters.REG_LAMDA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    train_model(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, num_epochs=parameters.NUM_EPOCHS, patience=4,reg_lamda=parameters.REG_LAMDA)
