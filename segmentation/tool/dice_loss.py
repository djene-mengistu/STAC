import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]

        # Apply softmax to the logits to get probabilities
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Calculate Dice coefficient
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs + targets_one_hot, dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        # Average over the batch and classes
        dice_loss = dice_loss.mean()

        return dice_loss

# Example usage:
# logits: output from your model with shape [B, C, H, W]
# targets: ground truth with shape [B, H, W] where values are class indices (0 to C-1)
# logits = torch.randn(4, 4, 224, 224)  # Example logits
# targets = torch.randint(0, 4, (4, 224, 224))  # Example target

# criterion = DiceLoss()
# loss = criterion(logits, targets)
# print(f"Dice Loss: {loss.item()}")
