import open3d.core as o3c
import numpy as np
import torch
import torch.utils.dlpack

# From PyTorch
th_a = torch.ones((5,)).cuda(0)
o3_a = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_a))
print(f"th_a: {th_a}")
print(f"o3_a: {o3_a}")
print("")

# Changes to PyTorch array reflects on open3d Tensor and vice versa
th_a[0] = 100
o3_a[1] = 200
print(f"th_a: {th_a}")
print(f"o3_a: {o3_a}")

# To PyTorch
o3_a = o3c.Tensor([1, 1, 1, 1, 1], device=o3c.Device("CUDA:0"))
th_a = torch.utils.dlpack.from_dlpack(o3_a.to_dlpack())
o3_a = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_a))
print(f"th_a: {th_a}")
print(f"o3_a: {o3_a}")
print("")

# Changes to PyTorch array reflects on open3d Tensor and vice versa
th_a[0] = 100
o3_a[1] = 200
print(f"th_a: {th_a}")
print(f"o3_a: {o3_a}")