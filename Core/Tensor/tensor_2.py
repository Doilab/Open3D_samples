import open3d.core as o3c
import numpy as np

''' binary_element-wise_operation '''
a = o3c.Tensor([1, 1, 1], dtype=o3c.Dtype.Float32)
b = o3c.Tensor([2, 2, 2], dtype=o3c.Dtype.Float32)
print("a + b = {}".format(a + b))
print("a - b = {}".format(a - b))
print("a * b = {}".format(a * b))
print("a / b = {}".format(a / b))

# Automatic broadcasting.
a = o3c.Tensor.ones((2, 3), dtype=o3c.Dtype.Float32)
b = o3c.Tensor.ones((3,), dtype=o3c.Dtype.Float32)
print("a + b = \n{}\n".format(a + b))

# Automatic type casting.
a = a[0]
print("a + 1 = {}".format(a + 1))  # Float + Int -> Float.
print("a + True = {}".format(a + True))  # Float + Bool -> Float.

# Inplace.
a -= True
print("a = {}".format(a))

''' unary_element-wisd_operation '''
a = o3c.Tensor([4, 9, 16], dtype=o3c.Dtype.Float32)
print("a = {}\n".format(a))
print("a.sqrt = {}\n".format(a.sqrt()))
print("a.sin = {}\n".format(a.sin()))
print("a.cos = {}\n".format(a.cos()))

# Inplace operation
a.sqrt_()
print(a)

''' reduction '''
vals = np.array(range(24)).reshape((2, 3, 4))
a = o3c.Tensor(vals)
print("a.sum = {}\n".format(a.sum()))
print("a.min = {}\n".format(a.min()))
print("a.ArgMax = {}\n".format(a.argmax()))

# With specified dimension.
vals = np.array(range(24)).reshape((2, 3, 4))
a = o3c.Tensor(vals)

print("Along dim=0\n{}".format(a.sum(dim=(0))))
print("Along dim=(0, 2)\n{}\n".format(a.sum(dim=(0, 2))))

# Retention of reduced dimension.
print("Shape without retention : {}".format(a.sum(dim=(0, 2)).shape))
print("Shape with retention : {}".format(a.sum(dim=(0, 2), keepdim=True).shape))

''' slicing, indexing, getitem, and setitem '''
vals = np.array(range(24)).reshape((2, 3, 4))
a = o3c.Tensor(vals)
print("a = \n{}\n".format(a))

# Indexing __getitem__.
print("a[1, 2] = {}\n".format(a[1, 2]))

# Slicing __getitem__.
print("a[1:] = \n{}\n".format(a[1:]))

# slice object.
print("a[:, 0:3:2, :] = \n{}\n".format(a[:, 0:3:2, :]))

# Combined __getitem__
print("a[:-1, 0:3:2, 2] = \n{}\n".format(a[:-1, 0:3:2, 2]))

vals = np.array(range(24)).reshape((2, 3, 4))
a = o3c.Tensor(vals)

# Changes get reflected.
b = a[:-1, 0:3:2, 2]
b[0] += 100
print("b = {}\n".format(b))
print("a = \n{}".format(a))

vals = np.array(range(24)).reshape((2, 3, 4))
a = o3c.Tensor(vals)

# Example __setitem__
a[:, :, 2] += 100
print(a)

''' Advanced indexing '''
vals = np.array(range(24)).reshape((2, 3, 4))
a = o3c.Tensor(vals)

# Along each dimension, a specific element is selected.
print("a[[0, 1], [1, 2], [1, 0]] = {}\n".format(a[[0, 1], [1, 2], [1, 0]]))

# Changes not reflected as it is a copy.
b = a[[0, 0], [0, 1], [1, 1]]
b[0] += 100
print("b = {}\n".format(b))
print("a[[0, 0], [0, 1], [1, 1]] = {}".format(a[[0, 0], [0, 1], [1, 1]]))

''' Combining advanced and basic indexing '''
vals = np.array(range(24)).reshape((2, 3, 4))
a = o3c.Tensor(vals)

print("a[1, 0:2, [1, 2]] = \n{}\n".format(a[1, 0:2, [1, 2]]))

# Subtle difference in selection and advanced indexing.
print("a[(0, 1)] = {}\n".format(a[(0, 1)]))
print("a[[0, 1] = \n{}\n".format(a[[0, 1]]))

a = o3c.Tensor(np.array(range(120)).reshape((2, 3, 4, 5)))

# Interleaving slice and advanced indexing.
print("a[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]] = \n{}\n".format(
    a[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]]))

''' Boolean array indexing '''
a = o3c.Tensor(np.array([1, -1, -2, 3]))
print("a = {}\n".format(a))

# Add constant to all negative numbers.
a[a < 0] += 20
print("a = {}\n".format(a))

''' Logical operations '''
a = o3c.Tensor(np.array([True, False, True, False]))
b = o3c.Tensor(np.array([True, True, False, False]))

print("a AND b = {}".format(a.logical_and(b)))
print("a OR b = {}".format(a.logical_or(b)))
print("a XOR b = {}".format(a.logical_xor(b)))
print("NOT a = {}\n".format(a.logical_not()))

# Only works for boolean tensors.
print("a.any = {}".format(a.any()))
print("a.all = {}\n".format(a.all()))

# If tensor is not boolean, 0 will be treated as False, while non-zero as true.
# The tensor will be filled with 0 or 1 casted to tensor's dtype.
c = o3c.Tensor(np.array([2.0, 0.0, 3.5, 0.0]))
d = o3c.Tensor(np.array([0.0, 3.0, 1.5, 0.0]))
print("c AND d = {}".format(c.logical_and(d)))

''' Comparision Operations '''
a = o3c.Tensor([0, 1, -1])
b = o3c.Tensor([0, 0, 0])

print("a > b = {}".format(a > b))
print("a >= b = {}".format(a >= b))
print("a < b = {}".format(a < b))
print("a <= b = {}".format(a <= b))
print("a == b = {}".format(a == b))
print("a != b = {}".format(a != b))

# Throws exception if device/dtype is not shape.
# If shape is not same, then tensors should be broadcast compatible.
print("a > b = {}".format(a > b[0]))

''' Nonzero operations '''
a = o3c.Tensor([[3, 0, 0], [0, 4, 0], [5, 6, 0]])

print("a = \n{}\n".format(a))
print("a.nonzero() = \n{}\n".format(a.nonzero()))
print("a.nonzero(as_tuple = 1) = \n{}".format(a.nonzero(as_tuple=1)))
