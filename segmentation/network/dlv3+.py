
# import sys
# sys.path.append('./')
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from segmentation_models_pytorch import DeepLabV3Plus #Install the modules of segmentation model
from thop import profile

model = DeepLabV3Plus("resnet18", encoder_weights="imagenet", classes=4, activation=None) #Change accordingly

# # print(model)
# # model = model.cuda()
# # model.eval()
# # print(model)
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {num_params}")

# # # MACs and FLOPs
# input_image = torch.randn(1, 3, 224, 224)
# macs, params = profile(model, inputs=(input_image,))
# print(f"MACs: {macs / 1e9} GMacs")
# print(f"FLOPs: {2 * macs / 1e9} GFLOPs")  # Each MAC counts as 2 FLOPs

# import time
# model = model.cuda()
# model.eval()

# batch_size = 1
# input_shape = (3, 224, 224)  # Change if needed
# dummy_input = torch.randn(batch_size, *input_shape).cuda()

# # # Warm-up (important for accurate timing on GPU)
# # with torch.no_grad():
# #     for _ in range(10):
# #         _ = model(dummy_input)

# # # Timed run
# # n_iterations = 1000  # Number of inference passes
# # start_time = time.time()

# # with torch.no_grad():
# #     for _ in range(n_iterations):
# #         _ = model(dummy_input)

# # end_time = time.time()
# # total_time = end_time - start_time
# # avg_time_per_image = total_time / (n_iterations * batch_size)
# # fps = 1.0 / avg_time_per_image
# # Warm-up
# with torch.no_grad():
#     for _ in range(10):
#         _ = model(dummy_input)

# # Timed run
# torch.cuda.synchronize()  # Ensure GPU is ready
# start_time = time.time()

# with torch.no_grad():
#     for _ in range(1000):
#         _ = model(dummy_input)

# torch.cuda.synchronize()  # Wait for GPU to finish
# end_time = time.time()

# # Calculate FPS
# total_time = end_time - start_time
# avg_time_per_image = total_time / (1000 * batch_size)
# fps = 1.0 / avg_time_per_image

# # Print results
# print(f"Total Inference Time: {total_time:.2f} seconds")
# print(f"Average Time per Image: {avg_time_per_image:.4f} seconds")
# print(f"FPS: {fps:.2f}")

# # Get peak memory usage in MB
# # peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
# # print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
# memory_usage = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
# peak_memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # Peak memory in MB
# print(f"Current GPU Memory Usage: {memory_usage:.2f} MB")
# print(f"Peak GPU Memory Usage: {peak_memory_usage:.2f} MB")

# # print(f"Average inference time: {avg_time_per_image * 1000:.2f} ms")
# # print(f"FPS: {fps:.2f}")
