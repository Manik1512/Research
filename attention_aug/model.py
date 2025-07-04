import torch
import torch.nn as nn
from torchvision import models
from utils.losses import *  
from utils.augment import *
from torchvision.models import EfficientNet_B0_Weights

import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score



class my_model(nn.Module):
    def __init__(self,attention_heads,batch_size,threshold,lambda_reg=1):
        super().__init__()
        self.BCE=nn.CrossEntropyLoss() # ager yeh use kerenge to sigmoid ki zarurat nahi hai model mai but during inference time , we use sigmoid 
        self.regul_loss=AttentionRegularizationLoss(attention_heads, 1280) # M,C
        self.lambda_reg = lambda_reg

        self.feature_extractor = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # Load EfficientNet-B0 with pretrained weights

        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        for idx, block in enumerate(self.feature_extractor.features):
            if idx < 5:  # freeze blocks 0 to 3
                for param in block.parameters():
                    param.requires_grad = False
            
        self.num_features = self.feature_extractor.classifier[1].in_features
        
        self.attention_heads=attention_heads
        self.feature_extractor.classifier = nn.Identity()
        self.attention_layer=attention(1280,attention_heads)
        self.threshold=threshold
        self.BAP_layer=BAPModule()
        self.classifier = nn.Linear(attention_heads * self.num_features, 1)
        self.attention_augment=attention_augment(batch_size,attention_heads,threshold)

        self.dropout = nn.Dropout(p=0.5)
    def forward(self,input):
        # if self.training:
            #raw image per forward pass 
            features = self.feature_extractor.features(input) # [B, C, H, W]
            attention_maps = self.attention_layer(features) #Ak deta hai yeh => [B, M, H, W] 
            pooled_features = self.BAP_layer(features,attention_maps)#=>fk => [B, M, C]
            B, M, C = pooled_features.shape
            flattened_features = pooled_features.view(B, M * C)  # Flatten the features=>P
            flattened_features = self.dropout(flattened_features)
            logits = self.classifier(flattened_features) # [B, num_classes]


            augmented_input=self.attention_augment.forward(attention_maps, input,task="train")   # [B, C, H, W]

            #augmented batch per forward pass
            augmented_features = self.feature_extractor.features(augmented_input)  # [B, C, H, W]
            augmented_attention_maps = self.attention_layer(augmented_features)  # [B, M, H, W]
            augmented_pooled_features = self.BAP_layer(augmented_features, augmented_attention_maps)  # [B, M, C]
            augmented_flattened_features = augmented_pooled_features.view(B, M * C)
            augmented_flattened_features = self.dropout(augmented_flattened_features)
            augmented_logits = self.classifier(augmented_flattened_features)


            return logits,augmented_logits, pooled_features
        
        # else:
    def predict(self, input):
        features = self.feature_extractor.features(input) # [B, C, H, W]
        attention_maps = self.attention_layer(features) #Ak deta hai yeh => [B, M, H, W] 
        pooled_features = self.BAP_layer(features,attention_maps)#=>fk => [B, M, C]
        B, M, C = pooled_features.shape
        flattened_features = pooled_features.view(B, M * C)  # Flatten the features=>P
        logits = self.classifier(flattened_features) # [B, num_classes]

        augmented_input=self.attention_augment.forward(attention_maps, input,task="val")

        augmented_features = self.feature_extractor.features(augmented_input)  # [B, C, H, W]
        augmented_attention_maps = self.attention_layer(augmented_features)  # [B, M, H, W]
        augmented_pooled_features = self.BAP_layer(augmented_features, augmented_attention_maps)  # [B, M, C]
        augmented_flattened_features = augmented_pooled_features.view(B, M * C)
        augmented_logits = self.classifier(augmented_flattened_features)
        return logits,augmented_logits, attention_maps, augmented_attention_maps,pooled_features



    

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
  


    attent=attention(input_channels=1280, attendtion_heads=5)
    Bap=BAPModule()


    # features= torch.randn(8,1280, 7, 7)  # Example input tensor
    # attention_maps = attent.forward(features=features)
    # pooled_features = Bap.forward(features, attention_maps)
    # print(f"shape of features=>{attention_maps.shape}")
    # print(f"shape of pooled features=>{pooled_features.shape}")

    # model=my_model(attention_heads=5,batch_size=8,threshold=0.5)
    # logits,augmented_logits, attention_maps, augmented_attention_maps, pooled_features = model(input=torch.randn(8,3,224,224)) 

    # print(f"shape of logits=>{logits.shape}")
    # print(f"shape of augmented_logits=>{augmented_logits.shape}")   
    # print(f"shape of attention_maps=>{attention_maps.shape}")
    # print(f"shape of augmented_attention_maps=>{augmented_attention_maps.shape}")
    # print(f"shape of pooled_features=>{pooled_features.shape}")   


    model=models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # Load EfficientNet-B0 with pretrained weights
    idx=0
    for idx, block in enumerate(model.features):
        if idx == 4:  
             break
        
    print(idx)