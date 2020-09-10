import time
import random
import os
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
#                                                                 SCALE_TO_range
# ==============================================================================
def scale_to_range(a, min, max, dtype=np.float):

    return ((a - min) / float(max - min)).astype(dtype)

# ==============================================================================
#                                                                 range_TO_SCALE
# ==============================================================================
def range_to_scale(a, min, max, dtype=np.float):

    return (a*(max -min)/min).astype(dtype)

# ==============================================================================
#                                                           POINT_CLOUD_TO_VOXEL
# ==============================================================================
def point_cloud_to_voxel(points,
                            x_res = 0.2,
                            y_res = 0.2,
                            z_res = 0.25,
                            x_height = (-40, 40),
                            y_height = (-80, 80),
                            z_height = (-3, 1),
                            d_range  = (0.1,80)
                            ):
    print("original point number {}".format(points.shape))
    points = points[ points[:,0] >  x_height[0] ]
    points = points[ points[:,0] <  x_height[1] ]
    points = points[ points[:,1] >  y_height[0] ]
    points = points[ points[:,1] <  y_height[1] ]
    points = points[ points[:,2] < -z_height[0] ]
    points = points[ points[:,2] > -z_height[1] ]
    print("afterslice point number {}".format(points.shape))
    # seperate axis 
    x_points = points[:, 0] - x_height[0]
    y_points = points[:, 1] - y_height[0]
    z_points = -points[:, 2] - z_height[0]
    d_points = np.sqrt(x_points ** 2 + y_points ** 2 + points[:,2]**2 )  # map distance relative to origin
    # calculate max range unit in meters
    x_height_total = -x_height[0] + x_height[1]
    y_height_total = -y_height[0] + y_height[1]
    z_height_total = -z_height[0] + z_height[1]
    # max pixel length in every axis 
    x_max = int(np.ceil(x_height_total / x_res))
    y_max = int(np.ceil(y_height_total / y_res))
    z_max = int(np.ceil(z_height_total / z_res))
    # compute img in every axis
    x_img = x_points / x_res
    x_img = np.trunc(x_img).astype(np.int32)
    y_img = y_points / y_res
    y_img = np.trunc(y_img).astype(np.int32)
    z_img = z_points / z_res
    z_img = np.trunc(z_img).astype(np.int32)
    # CONVERT TO IMAGE ARRAY
    img = np.zeros([z_max, y_max , x_max ], dtype=np.float)
    print(img.shape)
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])
#    img[z_img , y_img, x_img] = scale_to_1(d_points, min=d_range[0], max=d_range[1],dtype=np.float)
    img[z_img , y_img, x_img] = 1
    return img


# ==============================================================================
#                                                           VOXEL_TO_POINT_CLOUD
# ==============================================================================
def voxel_to_point_cloud(image,
                        x_res = 0.2,
                        y_res = 0.2,
                        z_res = 0.25,         #
                        x_height = (-40, 40), # 
                        y_height = (-80, 80), #
                        z_height = (-3, 1),   #
                        d_range = (0.1,80)    #
                        ):
    # RESOLUTION AND FIELD OF VIEW SETTINGS
    x_height_total = -x_height[0] + x_height[1]
    y_height_total = -y_height[0] + y_height[1]
    z_height_total = -z_height[0] + z_height[1]

    points = []
    pixel_values = []
    element = np.nditer(image, flags=['multi_index'])
    while not element.finished:
        if element.value != 0:
            pixel_values.append(element.value)
            points.append((element.multi_index[0],element.multi_index[1],element.multi_index[2]))
        element.iternext()
    points = np.array(points)
    z_image = points[:,0]
    y_image = points[:,1]
    x_image = points[:,2]

    pixel_values = np.array(pixel_values)
    z_points = z_image * z_res + z_height[0]
    y_points = y_image * y_res + y_height[0]
    x_points = x_image * x_res + x_height[0]

    points = np.array([x_points, y_points, z_points])
    points = points.T
    print(points.shape)
    return points


# ==============================================================================
#                                                           SLICE_TO_POINT_CLOUD
# ==============================================================================
def slice_to_point_cloud(image,
                        x_res = 0.2,
                        y_res = 0.2,
                        x_height = (-40, 40), # 
                        y_height = (-80, 80), #
                        d_range = (0.1,80)    #
                        ):
    # RESOLUTION AND FIELD OF VIEW SETTINGS
    x_height_total = -x_height[0] + x_height[1]
    y_height_total = -y_height[0] + y_height[1]

    points = []
    pixel_values = []
    element = np.nditer(image, flags=['multi_index'])
    while not element.finished:
        if element.value != 0:
            pixel_values.append(element.value)
            points.append((element.multi_index[0],element.multi_index[1]))
        element.iternext()
    points = np.array(points)
    x_image = points[:,0]
    y_image = points[:,1]

    pixel_values = np.array(pixel_values)
    y_points = y_image * y_res + y_height[0]
    x_points = x_image * x_res + x_height[0]

    points = np.array([x_points, y_points, pixel_values])
    points = points.T
    print(points.shape)
    return points


def remove_ground(o3d_points):
    plane_model, inliers = o3d_points.segment_plane(distance_threshold=0.1,
                                            ransac_n= 100,
                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    if abs(a)<0.05 and abs(b)<0.05 and abs(c-1) <0.05 and d<-0.25:
        print("Ground plane found")
        inlier_cloud = o3d_points.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 1])
        outlier_cloud = o3d_points.select_by_index(inliers, invert=True)
        outlier_cloud.paint_uniform_color([0, 1, 0])
        o3d_points = outlier_cloud
        # o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])
    return o3d_points

i=12
gt64 = "/home/buaaren/lidar-sr-dataset/gt64"
gt64_files = sorted(os.listdir(gt64))

if __name__ == "__main__":

    print("Running testing utils.py")

    pointcloud = np.fromfile(str("/home/buaaren/zeno/kitti_object/velodyne/training/velodyne/000425.bin"), dtype=np.float32, count=-1).reshape([-1,4])
    pointcloud = pointcloud[:,0:3]
    # gt = np.load(os.path.join(gt64,gt64_files[i]))
    # print(pointcloud.shape)

    voxel = point_cloud_to_voxel(pointcloud)
    gt = voxel_to_point_cloud(voxel)

    count_image = np.zeros((voxel.shape[1],voxel.shape[2]))
    height_image = np.zeros((voxel.shape[1],voxel.shape[2]))
    for i in range(voxel.shape[0]):
        slice = voxel[i,:,:]
        count_image += slice
        height_image += slice * (i+1)

    point_h = slice_to_point_cloud(height_image)
    point_n = slice_to_point_cloud(count_image)
    points = np.c_[ point_h, point_n[:,2] ]
    print(points.shape)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points=o3d.utility.Vector3dVector(pointss)
    # o3d.visualization.draw_geometries([pcd])

    # pcd = pcd.voxel_down_sample(voxel_size=0.2)

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(pcd.cluster_dbscan(eps=0.4, min_points=10, print_progress=True))

    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])


    # nbrs = NearestNeighbors(radius=5.0, algorithm='ball_tree', n_jobs=1,).fit(gt)
    # indices = nbrs.radius_neighbors(gt, return_distance=False)
    # # distances, indices = nbrs.kneighbors(gt)
    # # res = nbrs.kneighbors_graph(gt).toarray() # generate global adj matrix
    # # print(res.shape)
    # print(len(indices))
    # graphs = []
    # for indice in indices:
    #     graph = []
    #     for i in indice:
    #         graph.append(gt[i])
    #     graph = np.array(graph)
    #     pt1=o3d.geometry.PointCloud()
    #     pt1.points=o3d.utility.Vector3dVector(graph.reshape(-1,3))
    #     # pt1.paint_uniform_color(np.random.rand[1,3])
    #     graphs.append(pt1)
    
    # graphs = np.array(graphs)
    # print(graphs.shape)
    # print(graphs)



