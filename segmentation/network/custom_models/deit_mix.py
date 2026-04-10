import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Block

class PatchMerging(nn.Module):
    """Patch merging layer to reduce spatial resolution and increase channels."""
    def __init__(self, input_dim, output_dim):
        super(PatchMerging, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2)  # Halves spatial dims
        self.norm = nn.LayerNorm(output_dim)  # Normalize over the output dimension
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        x = self.conv(x)  # (B, C', H/2, W/2)
        B, C_new, H_new, W_new = x.shape
        x = x.reshape(B, C_new, -1).permute(0, 2, 1)  # (B, H/2 * W/2, C')
        x = self.norm(x)
        return x, H_new, W_new

class DEITMIX(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(DEITMIX, self).__init__()
        
        # Load pretrained DEiT backbone
        self.deit = timm.create_model('deit_small_patch16_224', pretrained=pretrained)
        self.deit.head = nn.Identity()  # Remove classification head
        
        # Parameters
        self.embed_dim = 384  # DEiT-small embedding dim
        self.patch_size = 16  # Initial patch size
        
        # Define channel dimensions for each stage
        self.stage_dims = [self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 4]  # [384, 768, 1536, 1536]
        
        # Stage 1: Use pretrained DEiT blocks
        self.stage_blocks = nn.ModuleList([
            nn.Sequential(*self.deit.blocks[:3]),  # Stage 1: 384 dim
        ])
        
        # Stages 2-4: Create new transformer blocks with updated dimensions
        for dim in self.stage_dims[1:]:
            self.stage_blocks.append(nn.Sequential(
                *[Block(dim=dim, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.) for _ in range(3)]
            ))
        
        # Patch merging layers between stages
        self.mergers = nn.ModuleList([
            PatchMerging(self.stage_dims[0], self.stage_dims[1]),  # 14x14 → 7x7, 384 → 768
            PatchMerging(self.stage_dims[1], self.stage_dims[2]),  # 7x7 → 4x4, 768 → 1536
            PatchMerging(self.stage_dims[2], self.stage_dims[3])   # 4x4 → 2x2, 1536 → 1536
        ])
        
        # Feature Pyramid Network (FPN) for consistent channel size
        self.fpn_channels = 256
        self.fpn = nn.ModuleList([
            nn.Conv2d(self.stage_dims[0], self.fpn_channels, kernel_size=1),  # Stage 1: 384 → 256
            nn.Conv2d(self.stage_dims[1], self.fpn_channels, kernel_size=1),  # Stage 2: 768 → 256
            nn.Conv2d(self.stage_dims[2], self.fpn_channels, kernel_size=1),  # Stage 3: 1536 → 256
            nn.Conv2d(self.stage_dims[3], self.fpn_channels, kernel_size=1)   # Stage 4: 1536 → 256
        ])
        
        # UpperNet head: fuse and upsample
        self.head = nn.Sequential(
            nn.Conv2d(self.fpn_channels * 4, self.fpn_channels, kernel_size=3, padding=1),  # Fuse all scales
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.fpn_channels, num_classes, kernel_size=16, stride=16)  # Upsample to 224x224
        )
        
    def forward(self, x):
        # Input shape: (B, 3, 224, 224)
        B = x.size(0)
        
        # Patch embedding
        x = self.deit.patch_embed(x)  # (B, 196, 384)
        H, W = 224 // self.patch_size, 224 // self.patch_size  # 14x14
        x = self.deit.pos_drop(x + self.deit.pos_embed[:, 1:, :])  # Add positional embeddings
        
        # Hierarchical stages
        features = []
        for i, stage in enumerate(self.stage_blocks):
            x = stage(x)  # Process through transformer blocks
            feat = x.permute(0, 2, 1).reshape(B, -1, H, W)
            features.append(feat)
            if i < len(self.mergers):  # Apply patch merging for next stage
                x, H, W = self.mergers[i](x, H, W)
        
        # Project to common channel size with FPN
        fpn_feats = [self.fpn[i](feat) for i, feat in enumerate(features)]
        
        # Upsample all features to the largest resolution (14x14)
        target_size = fpn_feats[0].shape[2:]  # 14x14
        fpn_feats = [
            f if i == 0 else nn.functional.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for i, f in enumerate(fpn_feats)
        ]
        
        # Concatenate along channel dimension
        fused = torch.cat(fpn_feats, dim=1)  # (B, 256 * 4, 14, 14)
        
        # UpperNet head
        logits = self.head(fused)  # (B, num_classes, 224, 224)
        
        return logits

# Example usage
def test_model():
    num_classes = 4
    model = DEITMIX(num_classes=num_classes, pretrained=True)
    
    # Dummy input
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (2, 21, 224, 224)

if __name__ == "__main__":
    test_model()