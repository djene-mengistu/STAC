import torch
import torch.nn as nn
from functools import partial
from vision_transformer import VisionTransformer, _cfg
# from timm.models.registry import register_model
from timm.models import register_model
# from timm.models.layers import trunc_normal_, to_2tuple
from timm.layers import trunc_normal_, to_2tuple
import torch.nn.functional as F
from thop import profile

import math

__all__ = ['deit_small_STAC']

class STAC(VisionTransformer):
    def __init__(self, decay_parameter=0.996, input_size=224, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        self.representation = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1)  #Change the output dimension accordingly [64,128,256]
        )
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False) #Upsample the representation to input size
        self.representation.apply(self._init_weights) 

        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)
        print(self.training)
        self.decay_parameter=decay_parameter
         
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.num_patches
        if npatch == N and w == h:
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
            )

        # print("w0:", w0, "h0:", h0, "patch_pos_embed.shape:", patch_pos_embed.shape)
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
            x = x + pos_embed_pat
        else:
            x = x + self.pos_embed_pat

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        attn_weights = []
        class_embeddings = []
        att_feat = []

        for i, blk in enumerate(self.blocks):
            x, weights_i, feat = blk(x) # x has a shape of B*199*384 -->199 is patch_embedding + class_token (196 + 3)
            attn_weights.append(weights_i)
            class_embeddings.append(x[:, 0:self.num_classes])
            att_feat.append(feat) 

        return x, x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights, class_embeddings, att_feat

    def forward(self, x, saliency=True, return_att=False, n_layers=12, attention_type='fused'):
        w, h = x.shape[2:]
        ftm, x_cls, x_patch, attn_weights, all_x_cls, att_feat = self.forward_features(x) 
        
        '''#ftm...>B*199*384 
        #c_cls...>B*3*384
        #x_patch...>B*196*384
        #attn_weights...>[B*6*199*199].....>12 * B * H * N * N-->like 12*B*6*199*199 after Concatenation
        #all_x_cls...>[B*384*3]....for each of 12 layers...> 12*B*384*3 after concatenation
        #att_feat..> [B*199*384]....for each of 12 layers...> 12*B*199*384 after concatenation
        '''
        
        n, p, c = x_patch.shape # B*196*384
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c]) #B*14*14*384
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2]) #B*384*14*14
        x_patch = x_patch.contiguous()
        rep_input = x_patch
        x_patch = self.head(x_patch) #B*C*14*14
        x_patch_flattened = x_patch.view(x_patch.shape[0], x_patch.shape[1], -1).permute(0, 2, 1) #B*196*C

        sorted_patch_token, indices = torch.sort(x_patch_flattened, -2, descending=True)
        weights = torch.logspace(start=0, end=x_patch_flattened.size(-2) - 1,
                                  steps=x_patch_flattened.size(-2), base=self.decay_parameter).cuda()
        x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()

        x_cls_logits = x_cls.mean(-1)

        output = []
        output.append(x_cls_logits)  ##Change accordingly
        output.append(torch.stack(all_x_cls))
        output.append(x_patch_logits)

        if saliency is not None:            
            # #ftm ----> of shape B*199*384        
            att_feat_all = torch.stack(att_feat) #12*B*199*384
            att_feat_all = torch.mean(att_feat_all, dim = 0) # B*199*384
            att_feat_all_patch = att_feat_all[:,self.num_classes:] #B*196*384 #Change accordingly for the num of classes [3,5,6, 20 ...NEU_Seg, MTD, DAGM, VOC12]
            # feat_map_all_patch = ftm[:, 3:] #B*196*384
        
            # # #Reshape and change to shape of B*384*14*14
            r, s, t = att_feat_all_patch.shape
            j, k, l, m = x_patch.shape
            att_feat_shaped = att_feat_all_patch.permute(0,2,1).reshape(r, t, l,m)
            # # feat_map_shaped = feat_map_all_patch.permute(0,2,1).reshape(r, t, 14,14)

            # # feat_m1 = self.head(feat_map_shaped)
            # # feat_m1 = F.relu(feat_m1)
            feat_at = self.head(att_feat_shaped)
            feat_at = F.relu(feat_at)
                    
            # #INput to sal_head
            
            # feature_map = x_patch.detach().clone()  # B * C * 14 * 14 
            feature_map = F.relu(x_patch)
            n, c, h, w = feature_map.shape
            attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N-->like 12*B*6*199*199 
            attn_weights = torch.mean(attn_weights, dim=2) #12*B*199*199 
            mtatt = attn_weights[-n_layers:].mean(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w]) #B*C*14*14--->16*3*14*14             
            cams1 = mtatt*feature_map 
            cams2 =mtatt*feat_at #Optional 
        
        if return_att:
            feature_map = x_patch.detach().clone()  # B * C * 14 * 14
            feature_map = F.relu(feature_map)
            n, c, h, w = feature_map.shape

            attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N-->like 12*B*6*199*199
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * 199 * 199
            mtatt = attn_weights[-n_layers:].mean(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w]) #B*C*14*14
            patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:] # 12*B*196*196
            if attention_type == 'fused':
                cams = mtatt * feature_map  # B*C*14*14
                cams = torch.sqrt(cams)
            elif attention_type == 'patchcam':
                cams = feature_map #B*C*14*14 
            elif attention_type == 'mct':
                cams = mtatt
            else:
                raise f'Error! {attention_type} is not defined!'

            x_logits = (x_cls_logits + x_patch_logits) / 2
            return x_logits, cams, patch_attn
        else:
            proj = self.representation(rep_input) # representation features for the contrastive learning
            proj = self.upsample(proj) #Upsamling to the resolution of the input image

            return output , cams1, cams2, proj  

@register_model
def deit_small_STAC(pretrained=True, **kwargs):
    model = STAC(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True, #Change number of heads to 8 from 6
        norm_layer=partial(nn.LayerNorm, eps=1e-6)) #norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

# import time
# from torchinfo import summary
# model =  deit_small_STAC().cuda()
# model =  deit_medium_STAC().cuda()
# # print(model)
# img= torch.randn(2,3,224,224).cuda()
# Number of parameters
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {num_params}")

# # MACs and FLOPs
# input_image = torch.randn(1, 3, 224, 224).cuda()
# macs, params = profile(model, inputs=(input_image,))
# print(f"MACs: {macs / 1e9} GMacs")
# print(f"FLOPs: {2 * macs / 1e9} GFLOPs")  # Each MAC counts as 2 FLOPs
# batch_size = 1
# input_shape = (3, 224, 224)  # Change if needed
# dummy_input = torch.randn(batch_size, *input_shape).cuda()

# # Warm-up (important for accurate timing on GPU)
# with torch.no_grad():
#     for _ in range(10):
#         _ = model(dummy_input)

# # Timed run
# n_iterations = 1000  # Number of inference passes
# start_time = time.time()

# with torch.no_grad():
#     for _ in range(n_iterations):
#         _ = model(dummy_input)

# end_time = time.time()
# total_time = end_time - start_time
# avg_time_per_image = total_time / (n_iterations * batch_size)
# fps = 1.0 / avg_time_per_image

# # Get peak memory usage in MB
# # peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
# # print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
# memory_usage = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
# peak_memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # Peak memory in MB
# print(f"Current GPU Memory Usage: {memory_usage:.2f} MB")
# print(f"Peak GPU Memory Usage: {peak_memory_usage:.2f} MB")

# print(f"Average inference time: {avg_time_per_image * 1000:.2f} ms")
# print(f"FPS: {fps:.2f}")
# print(summary(model, input_size=(3, 224, 224)))
# print(summary(model, input_size=(1, 3, 224, 224), depth=3))
# out, cams1, cams2, proj= model(img)
# print(out[0].shape)
# print(out[1].shape)
# print(out[2].shape)
# print('cams1', cams1.shape)
# print('cams2', cams2.shape)
# print('PROJ', proj.shape)
# # print(cams1[0])
# print(out.shape)
# print(cams.shape)
# print(x.shape)


