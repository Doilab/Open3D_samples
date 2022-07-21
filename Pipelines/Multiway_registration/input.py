import open3d as o3d

def load_point_clouds(voxel_size=0.0):
    pcds = []
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in demo_icp_pcds.paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

voxel_size = 0.02
pcds_down = load_point_clouds(voxel_size)
o3d.visualization.draw_geometries(pcds_down,
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])