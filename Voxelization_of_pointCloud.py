#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

pointCloud = o3d.io.read_point_cloud('./data/toolbox.pcd')
o3d.visualization.draw_geometries([pointCloud])

voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pointCloud, 0.01)   # ボクセルのサイズ設定
o3d.visualization.draw_geometries([voxel])

voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pointCloud, 0.0005)   # ボクセルのサイズ設定
o3d.visualization.draw_geometries([voxel])
