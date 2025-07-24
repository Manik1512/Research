# import torch
# from transformers import CLIPModel, CLIPProcessor
# from PIL import Image
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# import numpy as np

# # Config
# img_path = "/home/manik/Downloads/extracted_00101/content/00101/faces2/frame_0009.jpg"
# # img_path = "/home/manik/Downloads/Screenshot from 2025-07-24 13-15-27.png"  # <-- change this to your image path

# masking_ratio = 0.7  # 80% masked, 20% kept


# model_id="openai/clip-vit-base-patch32"
# # model_id="openai/clip-vit-large-patch14-336"
# # Load model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CLIPModel.from_pretrained(model_id, output_attentions=True).to(device).eval()
# processor = CLIPProcessor.from_pretrained(model_id)

# # Load image
# image = Image.open(img_path).convert("RGB")
# inputs = processor(images=image, return_tensors="pt").to(device)
# image_tensor = inputs["pixel_values"]  # (1, 3, 224, 224)

# # Get attention maps
# with torch.no_grad():
#     vision_out = model.vision_model(pixel_values=image_tensor, output_attentions=True)
#     last_attn = vision_out.attentions[-1][0]  # (heads, tokens, tokens)
#     hidden_states = vision_out.last_hidden_state[0]  # (tokens, dim)

# # Compute semantic attention score A from CLS to patches
# A = last_attn[:, 0, 1:]  # shape: (heads, num_patches)
# A = A.mean(0)            # (num_patches,) - average over heads
# A = A / A.sum()          # normalize (optional)

# # Decide which patches to keep
# num_keep = int(len(A) * (1 - masking_ratio))
# topk_indices = torch.topk(A, num_keep).indices.cpu().numpy()  # keep highest scoring

# # Mask image by patches
# patch_size = 32
# num_patches = A.shape[0]
# grid_size = int(num_patches ** 0.5)  # 7x7 for ViT-B/32

# # Resize original image to 224x224 (what model sees)
# resized_image = image.resize((224, 224))
# resized_np = np.array(resized_image)

# # Create a mask over patches
# mask = np.zeros((grid_size, grid_size), dtype=bool)
# for idx in topk_indices:
#     row = idx // grid_size
#     col = idx % grid_size
#     mask[row, col] = True

# # Apply mask to image (black out masked regions)
# masked_img = resized_np.copy()
# for i in range(grid_size):
#     for j in range(grid_size):
#         if not mask[i, j]:
#             y0 = i * patch_size
#             x0 = j * patch_size
#             masked_img[y0:y0+patch_size, x0:x0+patch_size] = 0  # blackout

# # Show result
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(resized_image)
# plt.title("Original")

# plt.subplot(1, 2, 2)
# plt.imshow(masked_img)
# plt.title(f"Semantic Masked ({(1 - masking_ratio)*100:.0f}% kept)")
# plt.axis("off")
# plt.show()

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F

# Config
# img_path = "/home/manik/Downloads/Screenshot from 2025-07-24 13-48-51.png"  # <-- change this to your image path

img_path = "/home/manik/Downloads/extracted_00101/content/00101/faces2/frame_0015.jpg"
masking_ratio = 0.7  # e.g., 70% masked, 30% kept
model_id = "openai/clip-vit-base-patch32"

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_id).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_id)

# Load and process image
image = Image.open(img_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)
image_tensor = inputs["pixel_values"]  # shape: (1, 3, 224, 224)

# Forward pass through vision encoder to get hidden states
with torch.no_grad():
    vision_out = model.vision_model(pixel_values=image_tensor, output_attentions=False)
    hidden_states = vision_out.last_hidden_state[0]  # shape: (tokens, dim)

# Separate CLS token and patch tokens
z_cls = hidden_states[0:1]              # (1, dim)
patch_tokens = hidden_states[1:]        # (N, dim)
N = patch_tokens.shape[0]               # number of patches

# Access Q/K projections from last transformer block
final_block = model.vision_model.encoder.layers[-1]
Q_proj = final_block.self_attn.q_proj
K_proj = final_block.self_attn.k_proj
num_heads = final_block.self_attn.num_heads
head_dim = Q_proj.out_features // num_heads

# Project CLS and patch tokens
Q_cls = Q_proj(z_cls)        # (1, dim)
K_all = K_proj(patch_tokens) # (N, dim)

# Reshape into (heads, dim) and (heads, N, dim)
Q_cls_heads = Q_cls.view(num_heads, head_dim)  # (heads, head_dim)
K_all_heads = K_all.view(N, num_heads, head_dim).permute(1, 0, 2)  # (heads, N, head_dim)

# Custom attention: A = (1/N) * sum_n softmax(C/N * Q_cls · K_n^T)
C = Q_cls.shape[-1]
scaling = C / N

# Attention score: (heads, N)
scores = torch.einsum("hd,hnd->hn", Q_cls_heads, K_all_heads) * scaling
A = F.softmax(scores, dim=-1).mean(0)  # (N,) averaged across heads

# Masking logic: keep top (1 - masking_ratio) tokens
num_keep = int(N * (1 - masking_ratio))
topk_indices = torch.topk(A, num_keep).indices.cpu().numpy()

# Patch/image parameters
patch_size = 32
grid_size = int(N ** 0.5)  # should be 7 for ViT-B/32

# Resize original image to 224x224
resized_image = image.resize((224, 224))
resized_np = np.array(resized_image)

# Build patch mask
mask = np.zeros((grid_size, grid_size), dtype=bool)
for idx in topk_indices:
    row = idx // grid_size
    col = idx % grid_size
    mask[row, col] = True

# Apply patch mask (blackout unselected regions)
masked_img = resized_np.copy()
for i in range(grid_size):
    for j in range(grid_size):
        if not mask[i, j]:
            y0 = i * patch_size
            x0 = j * patch_size
            masked_img[y0:y0+patch_size, x0:x0+patch_size] = 0

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(resized_image)
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(masked_img)
plt.title(f"Custom Attention Masked ({(1 - masking_ratio)*100:.0f}% kept)")
plt.axis("off")
plt.tight_layout()
plt.show()
