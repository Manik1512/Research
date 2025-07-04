import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionRegularizationLoss(nn.Module):
    def __init__(self, num_parts, feature_dim, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('centers', torch.zeros(num_parts, feature_dim))

    # def forward(self, f_parts):
    #     """
    #     f_parts: [B, M, C]
    #     """
    #     B, M, C = f_parts.shape
    #     f_mean = f_parts.mean(dim=0)  # shape: [M, C]

    #     # === Key Change: Use a clone of centers to avoid touching the buffer ===
    #     centers_for_loss = self.centers.clone().detach()  # safe for graph
    #     loss = F.mse_loss(f_mean, centers_for_loss, reduction='sum') / M

    #     # === Safe in-place update ===
    #     with torch.no_grad():
    #         delta = f_mean.detach() - self.centers
    #         self.centers.add_(self.alpha * delta)

    #     return loss


    def forward(self, f_parts):
        B, M, C = f_parts.shape
        f_mean = f_parts.mean(dim=0)  # shape: [M, C]
        centers_for_loss = self.centers.clone().detach()
        loss = F.mse_loss(f_mean, centers_for_loss, reduction='sum') / M

        if self.training:  # ✅ Only update centers in training mode
            with torch.no_grad():
                delta = f_mean.detach() - self.centers
                self.centers.add_(self.alpha * delta)
        # else:
        #     print("Validation mode — skipping center update")
        return loss


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    loss_fn = AttentionRegularizationLoss(num_parts=4, feature_dim=1280).cuda()
    f_parts = torch.randn(8, 4, 1280, device='cuda', requires_grad=True)

    loss = loss_fn(f_parts)
    loss.backward()
    print("✅ Loss and backward successful")
