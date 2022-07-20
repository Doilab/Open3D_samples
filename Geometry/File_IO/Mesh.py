import open3d as o3d

print("Testing IO for meshes ...")
knot_data = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(knot_data.path)
print(mesh)
o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)