import open3d as o3d
import numpy as np
import copy

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_r = copy.deepcopy(mesh)
R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
mesh_r.rotate(R, center=(0, 0, 0))
o3d.visualization.draw_geometries([mesh, mesh_r])

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_r = copy.deepcopy(mesh).translate((2, 0, 0))
mesh_r.rotate(mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4)),
              center=(0, 0, 0))
o3d.visualization.draw_geometries([mesh, mesh_r])