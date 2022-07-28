import open3d.core as o3c
import numpy as np

''' Tensor creation '''
# Tensor from list.
a = o3c.Tensor([0, 1, 2])
print("Create from list:\n{}".format(a))

# Tensor from Numpy.
a = o3c.Tensor(np.array([0, 1, 2]))
print("\nCreated from numpy array:\n{}".format(a))

# Dtype and inferred from list.
a_float = o3c.Tensor([0.0, 1.0, 2.0])
print("\nDefault dtype and device:\n{}".format(a_float))

# Specify dtype.
a = o3c.Tensor(np.array([0, 1, 2]), dtype=o3c.Dtype.Float64)
print("\nSpecified data type:\n{}".format(a))

# Specify device. CUDA
a = o3c.Tensor(np.array([0, 1, 2]), device=o3c.Device("CUDA:0"))
print("\nSpecified device:\n{}".format(a))

# Shallow copy constructor.
vals = np.array([1, 2, 3])
src = o3c.Tensor(vals)
dst = src
src[0] += 10

# Changes in one will get reflected in other.
print("Source tensor:\n{}".format(src))
print("\nTarget tensor:\n{}".format(dst))

''' Properties of a tensor '''
vals = np.array((range(24))).reshape(2, 3, 4)
a = o3c.Tensor(vals, dtype=o3c.Dtype.Float64, device=o3c.Device("CUDA:0"))
print(f"a.shape: {a.shape}")
print(f"a.strides: {a.strides}")
print(f"a.dtype: {a.dtype}")
print(f"a.device: {a.device}")
print(f"a.ndim: {a.ndim}")


''' Copy & device transfer '''
# Host -> Device.
a_cpu = o3c.Tensor([0, 1, 2])
a_gpu = a_cpu.cuda(0)
print(a_gpu)

# Device -> Host.
# CUDA
a_gpu = o3c.Tensor([0, 1, 2], device=o3c.Device("CUDA:0"))
a_cpu = a_gpu.cpu()
print(a_cpu)

# Device -> another Device.
a_gpu_0 = o3c.Tensor([0, 1, 2], device=o3c.Device("CUDA:0"))
a_gpu_1 = a_gpu_0.cuda(0)
print(a_gpu_1)


''' Data types '''
# Type casting
# E.g. float -> int
a = o3c.Tensor([0.1, 1.5, 2.7])
b = a.to(o3c.Dtype.Int32)
print(a)
print(b)

# E.g. float -> int
a = o3c.Tensor([0.1, 1.5, 2.7])
b = a.to(o3c.Dtype.Int32)
print(a)
print(b)

# E.g. int -> float
a = o3c.Tensor([1, 2, 3])
b = a.to(o3c.Dtype.Float32)
print(a)
print(b)

''' Numpy I/O with direct memory map '''
# Using constructor.
np_a = np.ones((5,), dtype=np.int32)
o3_a = o3c.Tensor(np_a)
print(f"np_a: {np_a}")
print(f"o3_a: {o3_a}")
print("")

# Changes to numpy array will not reflect as memory is not shared.
np_a[0] += 100
o3_a[1] += 200
print(f"np_a: {np_a}")
print(f"o3_a: {o3_a}")

# From numpy.
np_a = np.ones((5,), dtype=np.int32)
o3_a = o3c.Tensor.from_numpy(np_a)

# Changes to numpy array reflects on open3d Tensor and vice versa.
np_a[0] += 100
o3_a[1] += 200
print(f"np_a: {np_a}")
print(f"o3_a: {o3_a}")

# To numpy.
o3_a = o3c.Tensor([1, 1, 1, 1, 1], dtype=o3c.Dtype.Int32)
np_a = o3_a.numpy()

# Changes to numpy array reflects on open3d Tensor and vice versa.
np_a[0] += 100
o3_a[1] += 200
print(f"np_a: {np_a}")
print(f"o3_a: {o3_a}")

# For CUDA Tensor, call cpu() before calling numpy().
o3_a = o3c.Tensor([1, 1, 1, 1, 1], device=o3c.Device("CUDA:0"))
print(f"\no3_a.cpu().numpy(): {o3_a.cpu().numpy()}")
