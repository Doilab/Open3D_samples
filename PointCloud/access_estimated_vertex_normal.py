import open3d as o3d
import numpy as np

print("Load p ply point cloud, print it, and read it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)

print("Print a normal venctor of the 0th point")
print(downpcd.normals[0])

print("Print a normal venctor of the first 10th point")
print(np.asarray(downpcd.normals)[:10, :])