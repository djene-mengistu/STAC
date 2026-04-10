import torch
torch.cuda.set_device(0) 
total_memory = torch.cuda.get_device_properties(1).total_memory 
allocated_memory = int(total_memory * 0.1) 
tmp_tensor = torch.empty(allocated_memory, dtype=torch.int8, device='cuda:0')