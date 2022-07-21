import open3d as o3d
import numpy as np
import copy

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])

print("1. Load two point clouds and show initial pose")
demo_colored_icp_pcds = o3d.data.DemoColoredICPPointClouds()
source = o3d.io.read_point_cloud(demo_colored_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_colored_icp_pcds.paths[1])

# draw initial alignment
current_transformation = np.identity(4)
draw_registration_result_original_color(source, target, current_transformation)
