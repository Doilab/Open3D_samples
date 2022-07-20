import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

''' 初期化 '''
cube = o3d.t.geometry.TriangleMesh.from_legacy(
    o3d.geometry.TriangleMesh.create_box().translate([-1.2, -1.2, 0]))
sphere = o3d.t.geometry.TriangleMesh.from_legacy(
    o3d.geometry.TriangleMesh.create_sphere(0.5).translate([0.7, 0.8, 0]))

scene = o3d.t.geometry.RaycastingScene()
# Add triangle meshes and remember ids
mesh_ids = {}
mesh_ids[scene.add_triangles(cube)] = 'cube'
mesh_ids[scene.add_triangles(sphere)] = 'sphere'

''' 表面上の最も近い点を計算する '''
query_point = o3d.core.Tensor([[0, 0, 0]], dtype=o3d.core.Dtype.Float32)

# We compute the closest point on the surface for the point at position [0,0,0].
ans = scene.compute_closest_points(query_point)

# Compute_closest_points provides the point on the surface, the geometry id,
# and the primitive id.
# The dictionary keys are
#.    points
#.    geometry_ids
#.    primitive_ids
print('The closest point on the surface is', ans['points'].numpy())
print('The closest point is on the surface of the',
      mesh_ids[ans['geometry_ids'][0].item()])
print('The closest point belongs to triangle', ans['primitive_ids'][0].item())

rays = np.concatenate(
    [query_point.numpy(),
     np.ones(query_point.shape, dtype=np.float32)],
    axis=-1)
intersection_counts = scene.count_intersections(rays).numpy()
# A point is inside if the number of intersections with the scene is even
# This sssumes that inside and outside is we ll defined for the scene.
is_inside = intersection_counts % 2 == 1

def compute_signed_distance_and_closest_goemetry(query_points: np.ndarray):
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    return distance, closest_points['geometry_ids'].numpy()

# compute range
xyz_range = np.linspace([-2, -2, -2], [2, 2, 2], num=32)
# query_points is a [32,32,32,3] array ..
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

sdf, closest_geom = compute_signed_distance_and_closest_goemetry(query_points)

# We can visualize a slice of the grids directly with matplotlib
fig, axes = plt.subplots(1, 2)
axes[0].imshow(sdf[:, :, 16])
axes[1].imshow(closest_geom[:, :, 16])