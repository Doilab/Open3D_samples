import open3d.core as o3c
import numpy as np

capacity = 10
device = o3c.Device('cpu:0')

''' A simple example '''
hashmap = o3c.HashMap(capacity,
                      key_dtype=o3c.int64,
                      key_element_shape=(1,),
                      value_dtype=o3c.int64,
                      value_element_shape=(1,),
                      device=device)

''' Insertion '''
# Prepare a batch of 7 key/values, each a int64 element
keys = o3c.Tensor([[100], [200], [400], [800], [300], [200], [100]],
                  dtype=o3c.int64,
                  device=device)
vals = o3c.Tensor([[1], [2], [4], [8], [3], [2], [1]],
                  dtype=o3c.int64,
                  device=device)
buf_indices, masks = hashmap.insert(keys, vals)

print('masks: \n', masks)
print('inserted keys: \n', keys[masks])

buf_keys = hashmap.key_tensor()
buf_vals = hashmap.value_tensor()
buf_indices = buf_indices[masks].to(o3c.int64)
print('buffer indices: \n', buf_indices)

inserted_keys = buf_keys[buf_indices]
inserted_vals = buf_vals[buf_indices]
print('inserted keys: \n', inserted_keys)
print('inserted values: \n', inserted_vals)

''' Query '''
print()
query_keys = o3c.Tensor([[1000], [100], [300], [200], [100], [0]],
                        dtype=o3c.int64,
                        device=device)
buf_indices, masks = hashmap.find(query_keys)
valid_keys = query_keys[masks]
buf_indices = buf_indices[masks].to(o3c.int64)
valid_vals = hashmap.value_tensor()[buf_indices]
print('found valid keys: \n', valid_keys)
print('found valid values: \n', valid_vals)

''' Active entries in the hash map '''
print()
def print_active_entries(hashmap):
    active_buf_indices = hashmap.active_buf_indices().to(o3c.int64)

    active_keys = hashmap.key_tensor()[active_buf_indices]
    print('all active keys:\n', active_keys)

    active_vals = hashmap.value_tensor()[active_buf_indices]
    print('all active values:\n', active_vals)

''' Erase '''
print()
erase_keys = o3c.Tensor([[100], [1000], [100]], dtype=o3c.int64, device=device)
masks = hashmap.erase(erase_keys)
print('erase masks:\n', masks)
print('erased keys:\n', erase_keys[masks])

print_active_entries(hashmap)

''' Activate '''
print()
activate_keys = o3c.Tensor([[1000], [0]], dtype=o3c.int64, device=device)
buf_indices, masks = hashmap.activate(activate_keys)

buf_vals = hashmap.value_tensor()
# Note the assigned tensor has to be strictly in the shape of (N, 1) due to broadcasting
buf_vals[buf_indices[masks].to(o3c.int64)] \
    = o3c.Tensor([[10], [0]],
                 dtype=o3c.int64,
                 device=device)

print_active_entries(hashmap)

''' Rehashing and reserve '''
print()
print('size:', hashmap.size())
print('capacity:', hashmap.capacity())

keys = o3c.Tensor([[700], [1200], [1500]], dtype=o3c.int64, device=device)
vals = o3c.Tensor([[7], [12], [-1]], dtype=o3c.int64, device=device)
buf_indices, masks = hashmap.insert(keys, vals)
print('size:', hashmap.size())
print('capacity:', hashmap.capacity())
print_active_entries(hashmap)

keys = o3c.Tensor([[1600], [1700], [1800]], dtype=o3c.int64, device=device)
vals = o3c.Tensor([[16], [17], [18]], dtype=o3c.int64, device=device)
buf_indices, masks = hashmap.insert(keys, vals)
print('size:', hashmap.size())
print('capacity:', hashmap.capacity())
print_active_entries(hashmap)

hashmap.reserve(100)
print('size:', hashmap.size())
print('capacity:', hashmap.capacity())

''' Multi-valued hash map '''
print()
mhashmap = o3c.HashMap(capacity,
                       key_dtype=o3c.int32,
                       key_element_shape=(3,),
                       value_dtypes=(o3c.uint8, o3c.float32),
                       value_element_shapes=((3,), (1,)),
                       device=device)
voxel_coords = o3c.Tensor([[0, 1, 0], [-1, 2, 3], [3, 4, 1]],
                          dtype=o3c.int32,
                          device=device)
voxel_colors = o3c.Tensor([[0, 255, 0], [255, 255, 0], [255, 0, 0]],
                          dtype=o3c.uint8,
                          device=device)
voxel_weights = o3c.Tensor([[0.9], [0.1], [0.3]],
                           dtype=o3c.float32,
                           device=device)
mhashmap.insert(voxel_coords, (voxel_colors, voxel_weights))

query_coords = o3c.Tensor([[0, 1, 0]], dtype=o3c.int32, device=device)
buf_indices, masks = mhashmap.find(query_coords)

valid_keys = query_coords[masks]
buf_indices = buf_indices[masks].to(o3c.int64)
valid_colors = mhashmap.value_tensor(0)[buf_indices]
valid_weights = mhashmap.value_tensor(1)[buf_indices]
print('found coordinates:\n', valid_keys)
print('found colors:\n', valid_colors)
print('found weights:\n', valid_weights)

def print_active_multivalue_entries(mhashmap):
    active_buf_indices = mhashmap.active_buf_indices().to(o3c.int64)

    active_keys = mhashmap.key_tensor()[active_buf_indices]
    print('all active keys:\n', active_keys)

    n_buffers = len(mhashmap.value_tensors())
    for i in range(n_buffers):
        active_val_i = mhashmap.value_tensor(i)[active_buf_indices]
        print('active value {}\n:'.format(i), active_val_i)


print_active_multivalue_entries(mhashmap)

''' Hash set '''
print()
hashset = o3c.HashSet(capacity,
                      key_dtype=o3c.int64,
                      key_element_shape=(1,),
                      device=device)
keys = o3c.Tensor([1, 3, 5, 7, 5, 3, 1], dtype=o3c.int64,
                  device=device).reshape((-1, 1))
hashset.insert(keys)

keys = o3c.Tensor([5, 7, 9, 11], dtype=o3c.int64, device=device).reshape(
    (-1, 1))
hashset.insert(keys)


def print_active_keys(hashset):
    active_buf_indices = hashset.active_buf_indices().to(o3c.int64)
    active_keys = hashset.key_tensor()[active_buf_indices]
    print('active keys:\n', active_keys)


print_active_keys(hashset)
