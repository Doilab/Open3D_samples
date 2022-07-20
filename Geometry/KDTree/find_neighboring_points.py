import open3d as o3d
import numpy as np

print("Testing kdtree in Open3D...")
print("Load a point cloud and paint it gray.")

sample_pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# [3]
print("Paint the 1501st point red.")
pcd.colors[1500] = [1, 0, 0]

# Using search_knn_vector_3d
print("Find its 200 nearest neighbors, and paint them blue.")
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

# Using search_radius_vector_3d
print("Find its neighbors with distance less than 0.2, and paint them green.")
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

print("Visualize the point cloud.")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.5599,
                                  front=[-0.4958, 0.8229, 0.2773],
                                  lookat=[2.1126, 1.0163, -1.8543],
                                  up=[0.1007, -0.2626, 0.9596])