import open3d as o3d

pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic("./camera_primesense.json")
print(pinhole_camera_intrinsic.intrinsic_matrix)