import open3d as o3d
import copy

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
print(f'Center of mesh: {mesh.get_center()}')
print(f'Center of mesh tx: {mesh_tx.get_center()}')
print(f'Center of mesh ty: {mesh_ty.get_center()}')
o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_mv = copy.deepcopy(mesh).translate((2, 2, 2), relative=False)
print(f'Center of mesh: {mesh.get_center()}')
print(f'Center of translated mesh: {mesh_mv.get_center()}')
o3d.visualization.draw_geometries([mesh, mesh_mv])