import open3d as o3d
import numpy as np

pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic("./camera_primesense.json")

redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()

source_color = o3d.io.read_image(redwood_rgbd.color_paths[0])
source_depth = o3d.io.read_image(redwood_rgbd.depth_paths[0])
target_color = o3d.io.read_image(redwood_rgbd.color_paths[1])
target_depth = o3d.io.read_image(redwood_rgbd.depth_paths[1])
source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    source_color, source_depth)
target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    target_color, target_depth)
target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    target_rgbd_image, pinhole_camera_intrinsic)

option = o3d.pipelines.odometry.OdometryOption()
odo_init = np.identity(4)
print(option)

[success_color_term, trans_color_term,
 info] = o3d.pipelines.odometry.compute_rgbd_odometry(
     source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init,
     o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
[success_hybrid_term, trans_hybrid_term,
 info] = o3d.pipelines.odometry.compute_rgbd_odometry(
     source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init,
     o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)