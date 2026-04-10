import torch
import torch.nn as nn
import torch.nn.functional as F
import network.resnet38d

class ContrastiveProjectionHead(nn.Module):
    """
    A lightweight projection head for dense (pixel-wise) contrastive learning.
    Takes a feature map of shape (B, in_channels, H, W) and outputs (B, out_channels, H, W).

    Args:
        in_channels (int): Number of input channels (e.g., 4096 from ResNet38).
        out_channels (int): Number of output channels. Default: 256.
    """
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.projection(x)


class Net(network.resnet38d.Net):
    def __init__(self, num_classes):
        super().__init__()

        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False)
        self.proj_head = ContrastiveProjectionHead(in_channels=4096, out_channels=256)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False) #Upsample the representation to input size       

        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8, self.proj_head, self.upsample]  # ← Add proj_head to from_scratch

    def forward(self, x, contrast=False):
        """
        Args:
            x (Tensor): Input image (B, 3, H, W)
            contrast (bool): If True, compute and return projection head output.
        Returns:
            If contrast=False: (pred, cam)
            If contrast=True:  (pred, cam, proj)
        """
        feat = super().forward(x)  # (B, 4096, H, W)
        cam = self.fc8(feat)       # (B, num_classes, H, W)

        # Global average pooling for classification logits
        h, w = cam.shape[-2:]
        pred = F.avg_pool2d(cam, kernel_size=(h, w)).view(cam.size(0), -1)

        if contrast:
            proj = self.proj_head(feat)  # (B, 256, H, W)
            proj = self.upsample(proj) #Upsamling to the resolution of the input image
            return pred, cam, proj
        else:
            return pred, cam

    def forward_cam(self, x):
        """Used during inference to get CAMs (no projection needed)."""
        feat = super().forward(x)
        cam = self.fc8(feat)
        return cam

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups