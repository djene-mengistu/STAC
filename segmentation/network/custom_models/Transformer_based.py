import torch
from torch import Tensor
from torch.nn import functional as F
import sys
sys.path.append('..')
from custom_models.base import BaseModel
from custom_models.upernet_head import UPerHead 
from custom_models.segformer_head import SegFormerHead
# from base import BaseModel
# from upernet_head import UPerHead 
# from segformer_head import SegFormerHead
from thop import profile

# from segmentation_models_pytorch import DeepLabV3Plus #Install the modules of segmentation model
# model = DeepLabV3Plus("resnet18", encoder_weights="imagenet", classes=4, activation=None) #Change accordingly
# # model = DeepLabV3Plus("resnet18", encoder_weights="imagenet", classes=4, activation=None) #Change accordingly
# print(model)
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {num_params}")

# # MACs and FLOPs
# input_image = torch.randn(1, 3, 224, 224)
# macs, params = profile(model, inputs=(input_image,))
# print(f"MACs: {macs / 1e9} GMacs")
# print(f"FLOPs: {2 * macs / 1e9} GFLOPs")  # Each MAC counts as 2 FLOPs

class Transformer_model(BaseModel):
    def __init__(self, backbone: str = 'MiT-B4', num_classes: int = 4) -> None: #Change the decoder type accordingly {PVT, MIT, ResT, and others}
        super().__init__(backbone, num_classes)
        # self.decode_head = UPerHead(self.backbone.channels, 128, num_classes) #For UpperNet head
        self.decode_head = SegFormerHead(self.backbone.channels, 128, num_classes) #For MLP head
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    net = Transformer_model('MiT-B4', 4)
    net.init_pretrained('/data/djene/djene/MCTCon/seg/network/custom_models/weights/mit_b4.pth')
    model = net
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print(model)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # MACs and FLOPs
    input_image = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input_image,))
    print(f"MACs: {macs / 1e9} GMacs")
    print(f"FLOPs: {2 * macs / 1e9} GFLOPs")  # Each MAC counts as 2 FLOPs
        

