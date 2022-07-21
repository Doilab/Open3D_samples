import open3d as o3d
import numpy as np
import ci

# pip install ci-py : is_ci

def load_fountain_dataset():
    rgbd_images = []
    fountain_rgbd_dataset = o3d.data.SampleFountainRGBDImages()
    for i in range(len(fountain_rgbd_dataset.depth_paths)):
        depth = o3d.io.read_image(fountain_rgbd_dataset.depth_paths[i])
        color = o3d.io.read_image(fountain_rgbd_dataset.color_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
        fountain_rgbd_dataset.keyframe_poses_log_path)
    mesh = o3d.io.read_triangle_mesh(
        fountain_rgbd_dataset.reconstruction_path)

    return mesh, rgbd_images, camera_trajectory

# Load dataset
mesh, rgbd_images, camera_trajectory = load_fountain_dataset()

# Run non-rigid optimization.
maximum_iteration = 100 if ci.is_ci else 300
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(
        mesh, rgbd_images, camera_trajectory,
        o3d.pipelines.color_map.NonRigidOptimizerOption(
            maximum_iteration=maximum_iteration))

o3d.visualization.draw_geometries([mesh],
                                  zoom=0.5399,
                                  front=[0.0665, -0.1107, -0.9916],
                                  lookat=[0.7353, 0.6537, 1.0521],
                                  up=[0.0136, -0.9936, 0.1118])