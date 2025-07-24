import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

class attention_augment(nn.Module):
    def __init__(self,batch_size, attention_heads,threshld):
        super().__init__()
        self.B= batch_size
        self.M= attention_heads
        self.threshold = threshld


    # def forward(self, attention_maps, raw_images):
    #     """
    #     Receives:
    #         attention_maps: [B, M, H, W] during training or [B, 1, H, W] during validation
    #         raw_images: [B, C, H, W]
    #     Applies either crop or drop based on one map per sample.
    #     """
    #     B = attention_maps.shape[0]
    #     M = attention_maps.shape[1] if attention_maps.dim() == 4 else 1  # handle [B, H, W]

    #     if attention_maps.dim() == 3:
    #         # Single map per sample [B, H, W] → [B, 1, H, W]
    #         attention_maps = attention_maps.unsqueeze(1)

    #     if M == 1:
    #         # Only one map per sample
    #         selected_maps = attention_maps[:, 0]  # [B, H, W]
    #     else:
    #         # Randomly select one map per sample
    #         indices = torch.randint(0, M, (B,), device=attention_maps.device)
    #         selected_maps = attention_maps[torch.arange(B, device=attention_maps.device), indices]  # [B, H, W]

    #     selected_maps = selected_maps.unsqueeze(1)  # [B, 1, H, W]

    #     # Random mode choice: 0 = drop, 1 = crop
    #     mode_flags = torch.randint(0, 2, (B,), device=attention_maps.device)

    #     augmented_images = []
    #     for i in range(B):
    #         if mode_flags[i] == 0:
    #             aug = self.attention_dropping(raw_images[i:i+1], selected_maps[i:i+1])
    #         else:
    #             aug = self.attention_cropping(raw_images[i:i+1], selected_maps[i:i+1])
    #         augmented_images.append(aug)

    #     return torch.cat(augmented_images, dim=0)  # [B, C, H, W]

    def forward(self,attention_maps,raw_images,task):
        """This recieves batch of maps , One map is randomly selected from each batch 
            The selected map is used to either drop or crop the image"""
        

        """problem-> in validation """
        augmented_images=[]
        if task=="train":
            self.B= attention_maps.shape[0]
            indices = torch.randint(0, self.M, (self.B,))  # Random index per sample
            selected_maps = torch.stack([attention_maps[i, idx] for i, idx in enumerate(indices)], dim=0)  # [B, H, W]=>harr batch sai eak map
            selected_maps=selected_maps.unsqueeze(1)

            # Random choice: 0 = drop, 1 = crop
            mode_flags = torch.randint(0, 2, (self.B,))  # one per sample
            
            for i in range(self.B):
                if mode_flags[i] == 0:
                    aug = self.attention_dropping(raw_images[i:i+1], selected_maps[i:i+1])  # keep dims
                else:
                    aug = self.attention_cropping(raw_images[i:i+1], selected_maps[i:i+1])
                    
                augmented_images.append(aug)


            return torch.cat(augmented_images, dim=0)  # shape: [B, C, H, W]
        else:
            self.B= attention_maps.shape[0]
            attention_avg = attention_maps.mean(dim=1, keepdim=True)#=>{ [B, 1, H, W]}
            for i in range(self.B):
                aug=self.attention_cropping(raw_images[i:i+1], attention_avg[i:i+1])
                augmented_images.append(aug)
            return torch.cat(augmented_images, dim=0)  # shape: [B, C, H, W]


    @torch.no_grad()
    def attention_dropping(self, input_image, attention_map, threshold=0.5):
        """
        Applies attention dropping using a single attention map for one image.

        Args:
            input_image (Tensor): [1, C, H_img, W_img]
            attention_map (Tensor): [1, 1, H_attn, W_attn]

        Returns:
            Tensor: Dropped image of shape [1, C, H_img, W_img]
        """
        _, C, H_img, W_img = input_image.shape
        _, _, H_attn, W_attn = attention_map.shape

        attn = attention_map[0, 0]

        # Normalize attention map
        attn_min, attn_max = attn.min(), attn.max()
        attn = (attn - attn_min) / (attn_max - attn_min + 1e-8)

        # Binarize to create drop mask (1 = keep, 0 = drop)
        drop_mask = (attn <= threshold).float()

        # Resize drop mask to match input image size
        drop_mask = F.interpolate(
            drop_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H_attn, W_attn]
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)  # [H_img, W_img]

        # Broadcast to all channels
        drop_mask = drop_mask.expand(C, H_img, W_img)

        # Apply drop mask
        dropped = input_image[0] * drop_mask
        return dropped.unsqueeze(0) # [1, C, H_img, W_img]


    @torch.no_grad()
    def attention_cropping(self, input_image, attention_map, threshold=0.5):
        """
        Applies attention cropping using a single attention map for one image.

        Args:
            input_image (Tensor): [1, C, H_img, W_img]
            attention_map (Tensor): [1, 1, H_attn, W_attn]

        Returns:
            Tensor: Cropped image resized back to [1, C, H_img, W_img]
        """
        _, C, H_img, W_img = input_image.shape
        _, _, H_attn, W_attn = attention_map.shape

        scale_y = H_img / H_attn
        scale_x = W_img / W_attn

        attn = attention_map[0, 0]

        # Normalize attention map
        attn_min, attn_max = attn.min(), attn.max()
        attn = (attn - attn_min) / (attn_max - attn_min + 1e-8)

        # Binarize using threshold
        mask = attn > threshold

        if not mask.any():
            # Fallback: center crop
            y1, y2 = H_img // 4, 3 * H_img // 4
            x1, x2 = W_img // 4, 3 * W_img // 4
        else:
            y_indices, x_indices = torch.where(mask)
            y1_attn, y2_attn = y_indices.min(), y_indices.max()
            x1_attn, x2_attn = x_indices.min(), x_indices.max()

            # Scale to image space
            y1 = int(y1_attn.item() * scale_y)
            y2 = int((y2_attn.item() + 1) * scale_y)
            x1 = int(x1_attn.item() * scale_x)
            x2 = int((x2_attn.item() + 1) * scale_x)

            # Clamp to image bounds
            y1 = max(0, min(H_img - 1, y1))
            y2 = max(y1 + 1, min(H_img, y2))  # ensure at least 1-pixel height
            x1 = max(0, min(W_img - 1, x1))
            x2 = max(x1 + 1, min(W_img, x2))  # ensure at least 1-pixel width

        # Crop and resize
        crop = input_image[0, :, y1:y2, x1:x2].unsqueeze(0)  # [1, C, h, w]
        crop = F.interpolate(
            crop, size=(H_img, W_img),
            mode='bilinear',
            align_corners=False,
            antialias=True
        )
        return crop  # [ 1,C, H_img, W_img]

# def show_sample(original, attn, augmented_image_list, idx=0):
#     fig, axs = plt.subplots(1, 3, figsize=(12, 3))

#     axs[0].imshow(original[idx].permute(1, 2, 0).numpy())
#     axs[0].set_title("Original")

#     axs[1].imshow(attn[idx, 0].numpy(), cmap='jet')
#     axs[1].set_title("Attention Map")

#     axs[2].imshow(augmented_image_list[idx].permute(1, 2, 0).numpy())
#     axs[2].set_title("Cropped")

#     for ax in axs:
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# def visualise_attention():
#     B, C, H, W = 8, 3, 256, 256
#     H_attn, W_attn = 16, 16

#     images = torch.zeros(B, C, H, W)

#     # Convert to NumPy to draw circles with OpenCV
#     for i in range(2):
#         img_np = np.zeros((H, W, 3), dtype=np.uint8)
#         if i == 0:
#             # Red circle in the center
#             cv2.circle(img_np, center=(W // 2, H // 2), radius=40, color=(255, 0, 0), thickness=-1)
#         elif i == 1:
#             # Green circle in the top-left
#             cv2.circle(img_np, center=(64, 64), radius=40, color=(0, 255, 0), thickness=-1)

#         # Convert to tensor and normalize to [0,1]
#         img_tensor = TF.to_tensor(img_np)  # shape: [3, H, W], range [0,1]
#         images[i] = img_tensor

#     # Attention maps: [B, 1, H_attn, W_attn]
#     attention_maps = torch.zeros(B, 1, H_attn, W_attn)

#     # Image 0: high attention in center
#     attention_maps[0, 0, 6:10, 6:10] = 1.0

#     # Image 1: high attention in top-left
#     attention_maps[1, 0, 0:4, 0:4] = 1.0

#     # Apply cropping
#     cropped = attention_dropping(images, attention_maps, threshold=0.5)

#     for i in range(2):
#         show_sample(images, attention_maps, cropped, idx=i)


if __name__ == "__main__":
    import numpy as np  
    import matplotlib.pyplot as plt
    import cv2
    import torchvision.transforms.functional as TF

    def show_sample(original, attn, augmented_image_list, idx=0):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))

        axs[0].imshow(original[idx].permute(1, 2, 0).numpy())
        axs[0].set_title("Original")

        axs[1].imshow(attn[idx, 0].numpy(), cmap='jet')
        axs[1].set_title("Attention Map")

        axs[2].imshow(augmented_image_list[idx].permute(1, 2, 0).numpy())
        axs[2].set_title("Cropped")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    def visualise_attention():
        B, C, H, W = 8, 3, 256, 256
        H_attn, W_attn = 16, 16

        images = torch.zeros(B, C, H, W)

        # Convert to NumPy to draw circles with OpenCV
        for i in range(2):
            img_np = np.zeros((H, W, 3), dtype=np.uint8)
            if i == 0:
                # Red circle in the center
                cv2.circle(img_np, center=(W // 2, H // 2), radius=40, color=(255, 0, 0), thickness=-1)
            elif i == 1:
                # Green circle in the top-left
                cv2.circle(img_np, center=(64, 64), radius=40, color=(0, 255, 0), thickness=-1)

            # Convert to tensor and normalize to [0,1]
            img_tensor = TF.to_tensor(img_np)  # shape: [3, H, W], range [0,1]
            images[i] = img_tensor

        # Attention maps: [B, 1, H_attn, W_attn]
        attention_maps = torch.zeros(B, 1, H_attn, W_attn)

        # Image 0: high attention in center
        attention_maps[0, 0, 6:10, 6:10] = 1.0

        # Image 1: high attention in top-left
        attention_maps[1, 0, 0:4, 0:4] = 1.0

        augment= attention_augment(batch_size=B, attention_heads=1, threshld=0.5)
        # Apply cropping
        cropped = augment.forward(attention_maps, images)

        for i in range(2):
            show_sample(images, attention_maps, cropped, idx=i)

    visualise_attention()