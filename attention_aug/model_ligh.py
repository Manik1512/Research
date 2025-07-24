import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
from torchvision import models
from utils.losses import *  
from utils.augment import *
from torchvision.models import EfficientNet_B0_Weights
from utils.metrics import EarlyStopping, log_metrics, compute_metrics, save_checkpoint
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torchmetrics as tm
# from torchmetrics import Metric
from pytorch_lightning.callbacks import EarlyStopping,Callback
from pytorch_lightning.loggers import TensorBoardLogger
from utils.data_pipeline import data_pipeline,train_transform,val_transform
logger=TensorBoardLogger("tb_logs", name="my_model")
import parameters
import os



def compute_total_loss(logits, augmented_logits, pooled_features, labels,lambda_reg,loss):
    criterion= nn.BCEWithLogitsLoss() 
    logits = torch.sigmoid(logits)
    augmented_logits = torch.sigmoid(augmented_logits)
    raw_loss = criterion(logits, labels)
    fine_loss = criterion(augmented_logits, labels)
    classification_loss = raw_loss + fine_loss
    regularization_loss = loss(pooled_features)
    total_loss = classification_loss + lambda_reg * regularization_loss
    
    return classification_loss , total_loss



# class my_model(pl.LightningModule):
#     def __init__(self,attention_heads,batch_size,threshold,lambda_reg=1):
#         super().__init__()
#         self.BCE=nn.CrossEntropyLoss() # ager yeh use kerenge to sigmoid ki zarurat nahi hai model mai but during inference time , we use sigmoid 
#         self.regul_loss=AttentionRegularizationLoss(attention_heads, 1280) # M,C
#         self.lambda_reg = lambda_reg

#         self.feature_extractor = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # Load EfficientNet-B0 with pretrained weights
#         self.criterion = nn.BCEWithLogitsLoss()
    

#         for idx, block in enumerate(self.feature_extractor.features):
#             if idx < 5:  # freeze blocks 0 to 3
#                 for param in block.parameters():
#                     param.requires_grad = False
            
#         self.num_features = self.feature_extractor.classifier[1].in_features
        
#         self.attention_heads=attention_heads
#         self.feature_extractor.classifier = nn.Identity()
#         self.attention_layer=attention(1280,attention_heads)
#         self.threshold=threshold
#         self.BAP_layer=BAPModule()
#         self.classifier = nn.Linear(attention_heads * self.num_features, 1)
#         self.attention_augment=attention_augment(batch_size,attention_heads,threshold)

#         self.dropout = nn.Dropout(p=0.5)
#         self.accuracy=tm.Accuracy(task="binary",threshold=0.5)
#         self.fi_score=tm.F1Score(task="binary",threshold=0.5)
#         self.precision=tm.Precision(task="binary",threshold=0.5)
#         self.recall=tm.Recall(task="binary",threshold=0.5)

#     def forward(self,input):
#         # if self.training:
#             #raw image per forward pass 
#             features = self.feature_extractor.features(input) # [B, C, H, W]
#             attention_maps = self.attention_layer(features) #Ak deta hai yeh => [B, M, H, W] 
#             pooled_features = self.BAP_layer(features,attention_maps)#=>fk => [B, M, C]
#             B, M, C = pooled_features.shape
#             flattened_features = pooled_features.view(B, M * C)  # Flatten the features=>P
#             flattened_features = self.dropout(flattened_features)
#             logits = self.classifier(flattened_features) # [B, num_classes]


#             augmented_input=self.attention_augment.forward(attention_maps, input,task="train")   # [B, C, H, W]

#             #augmented batch per forward pass
#             augmented_features = self.feature_extractor.features(augmented_input)  # [B, C, H, W]
#             augmented_attention_maps = self.attention_layer(augmented_features)  # [B, M, H, W]
#             augmented_pooled_features = self.BAP_layer(augmented_features, augmented_attention_maps)  # [B, M, C]
#             augmented_flattened_features = augmented_pooled_features.view(B, M * C)
#             augmented_flattened_features = self.dropout(augmented_flattened_features)
#             augmented_logits = self.classifier(augmented_flattened_features)


#             return logits,augmented_logits, pooled_features
    
#     def training_step(self, batch, batch_idx):
#         inputs, labels = batch
#         logits,augmented_logits, pooled_features = self.forward(inputs)
#         class_loss,loss = compute_total_loss(logits, augmented_logits, pooled_features, labels,lambda_reg=self.lambda_reg,loss=self.regul_loss)

#         return {"cls_loss":class_loss, "loss":loss}
    

#     def validation_step(self, batch, batch_idx):
#         inputs, labels = batch
#         logits, augmented_logits, pooled_features = self.forward(inputs)

#         P_raw = torch.sigmoid(logits)
#         P_fine_grained = torch.sigmoid(augmented_logits)
#         outputs = (P_raw + P_fine_grained) / 2.0

#         val_class_loss, val_loss_batch = compute_total_loss(
#             logits, augmented_logits, pooled_features, labels,
#             lambda_reg=self.lambda_reg, loss=self.regul_loss

#         )
#         # Store predictions and labels for step_end
#         return {
#             "outputs": outputs,
#             "labels": labels,
#             "val_class_loss": val_class_loss,
#             "val_loss_batch": val_loss_batch
#         }

    

#     def on_train_epoch_end(self, outputs):
#         class_loss=outputs["cls_loss"]
#         loss=outputs["loss"]
#         self.log_dict(class_loss=class_loss, loss=loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    
#     def on_validation_epoch_end(self, step_output):
#         outputs = step_output["outputs"]
#         labels = step_output["labels"]
#         val_class_loss = step_output["val_class_loss"]
#         val_loss_batch = step_output["val_loss_batch"]

#         # Update and log metrics
#         self.accuracy(outputs, labels)
#         self.fi_score(outputs, labels)
#         self.precision(outputs, labels)
#         self.recall(outputs, labels)

#         self.log_dict({
#             'val_accuracy': self.accuracy,
#             'val_f1_score': self.fi_score,
#             'val_precision': self.precision,
#             'val_recall': self.recall,
#             'val_class_loss': val_class_loss,
#             'val_loss_batch': val_loss_batch
#         }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': scheduler
#         }


class MyModel(pl.LightningModule):
    def __init__(self, attention_heads, batch_size, threshold, lambda_reg=1):
        super().__init__()

        self.save_hyperparameters()

        self.criterion = nn.BCEWithLogitsLoss()
        self.regul_loss = AttentionRegularizationLoss(attention_heads, 1280)
        self.lambda_reg = lambda_reg

        # Feature extractor
        self.feature_extractor = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for idx, block in enumerate(self.feature_extractor.features):
            if idx < 5:
                for param in block.parameters():
                    param.requires_grad = False

        self.num_features = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier = nn.Identity()

        self.attention_layer = attention(1280, attention_heads)
        self.BAP_layer = BAPModule()
        self.attention_augment = attention_augment(batch_size, attention_heads, threshold)
        self.classifier = nn.Linear(attention_heads * self.num_features, 1)
        self.dropout = nn.Dropout(p=0.5)

        self.accuracy = tm.Accuracy(task="binary", threshold=0.5)
        self.f1_score = tm.F1Score(task="binary", threshold=0.5)
        self.precision = tm.Precision(task="binary", threshold=0.5)
        self.recall = tm.Recall(task="binary", threshold=0.5)

        self.validation_outputs = []

    def forward(self, input):
        # Raw image forward pass
        features = self.feature_extractor.features(input)
        attention_maps = self.attention_layer(features)
        pooled_features = self.BAP_layer(features, attention_maps)
        B, M, C = pooled_features.shape
        flattened_features = self.dropout(pooled_features.view(B, M * C))
        logits = self.classifier(flattened_features)

        # Augmented image forward pass
        augmented_input = self.attention_augment(attention_maps, input, task="train")
        aug_features = self.feature_extractor.features(augmented_input)
        aug_attention_maps = self.attention_layer(aug_features)
        aug_pooled_features = self.BAP_layer(aug_features, aug_attention_maps)
        aug_flattened = self.dropout(aug_pooled_features.view(B, M * C))
        augmented_logits = self.classifier(aug_flattened)

        return logits, augmented_logits, pooled_features

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits, augmented_logits, pooled_features = self(inputs)
        class_loss, loss = compute_total_loss(logits, augmented_logits, pooled_features, labels,
                                              lambda_reg=self.lambda_reg, loss=self.regul_loss)

        self.log("train_cls_loss", class_loss, prog_bar=True)
        self.log("train_total_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits, augmented_logits, pooled_features = self(inputs)

        P_raw = torch.sigmoid(logits)
        P_fine = torch.sigmoid(augmented_logits)
        outputs = (P_raw + P_fine) / 2.0

        val_class_loss, val_loss_batch = compute_total_loss(
            logits, augmented_logits, pooled_features, labels,
            lambda_reg=self.lambda_reg, loss=self.regul_loss
        )

        self.validation_outputs.append({
            "outputs": outputs.detach(),
            "labels": labels.detach(),
            "val_class_loss": val_class_loss.detach(),
            "val_loss_batch": val_loss_batch.detach()
        })

    def on_validation_epoch_end(self):
        outputs = torch.cat([o["outputs"] for o in self.validation_outputs], dim=0)
        labels = torch.cat([o["labels"] for o in self.validation_outputs], dim=0)
        val_class_loss = torch.stack([o["val_class_loss"] for o in self.validation_outputs]).mean()
        val_loss_batch = torch.stack([o["val_loss_batch"] for o in self.validation_outputs]).mean()

        self.accuracy(outputs, labels)
        self.f1_score(outputs, labels)
        self.precision(outputs, labels)
        self.recall(outputs, labels)

        self.log_dict({
            'val_accuracy': self.accuracy,
            'val_f1_score': self.f1_score,
            'val_precision': self.precision,
            'val_recall': self.recall,
            'val_class_loss': val_class_loss,
            'val_loss_batch': val_loss_batch
        }, prog_bar=True)

        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



    

class BAPModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,features, attention_maps):
        maps=attention_maps.unsqueeze(2)# (B, M, 1, 7, 7)-> AK mai eak extra dim add ki hai 
        features = features.unsqueeze(1) # (B, 1, C, 7, 7)->F mai extra dim add ki 

        attended_features = features * maps # Fk (B, M, C, 7, 7)->Fk
        pooled_features = attended_features.mean(dim=[-2, -1])  # (B, M, C) ->fk =>GAP 

        return pooled_features


class attention(nn.Module): # checked
    
    def __init__(self,input_channels,attendtion_heads):
        super().__init__()
        self.input_size = input_channels
        self.conv1x1 = nn.Conv2d(input_channels, out_channels=attendtion_heads, kernel_size=1)
        self.BN=nn.BatchNorm2d(attendtion_heads)
        self.relu=nn.ReLU()
    def forward(self,features): 
        """Takes (B,C,H,W) and returns (B,M,H,W) where M is the number of attention heads"""
        attention_maps = self.conv1x1(features)
        attention_maps = self.BN(attention_maps)
        attention_maps = self.relu(attention_maps)
        return attention_maps
   

         
if __name__ == "__main__":
    model=MyModel(attention_heads=parameters.ATTENTION_HEADS,batch_size=parameters.BATCH_SIZE,threshold=0.5,lambda_reg=parameters.REG_LAMDA)

    train_loader,val_loader = data_pipeline(full_path=None,
                                            train_transform=train_transform,
                                            val_transform=val_transform,
                                            batch=parameters.BATCH_SIZE,
                                            val_already_split=True,
                                            train_path=os.path.join(parameters.DATASET_PATH, "Train"),
                                            val_path=os.path.join(parameters.DATASET_PATH, "Validation")
                                            )
    trainer=pl.Trainer(accelerator="gpu",min_epochs=10,max_epochs=20,precision="16-mixed",devices=1,callbacks=EarlyStopping(monitor='val_loss_batch', patience=3, mode='min'),logger=logger)
    trainer.fit(model,train_loader,val_loader)





   