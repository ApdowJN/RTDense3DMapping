import numpy as np
import sys
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
from tqdm import tqdm

def read_points(data_path, s=0.01):
    ply_list = []
    file_list = os.listdir(data_path)
    file_list.sort()

    for pc in tqdm(file_list, desc="Loading Point Clouds", unit="point-clouds"):
        if(pc[-3:] != "ply"):
            continue
        ply_path = os.path.join(data_path,pc)
        ply = o3d.io.read_point_cloud(ply_path)
        ply = ply.voxel_down_sample(voxel_size=s)

        ply_list.append(ply)

    return ply_list

def save_ply(ply, file_path):
    print("Saving merged point cloud...")
    o3d.io.write_point_cloud(file_path, ply)

def merge_clouds(ply_list):
    ply = o3d.geometry.PointCloud()

    for point_id in tqdm(range(len(ply_list)), desc="Merging Point Clouds", unit="clouds"):
        ply += ply_list[point_id]
    return ply

def remove_outliers(ply, neighbors, std):
    print("Removing outliers...")
    pre_filt_size = len(ply.points)

    cl, ind = ply.remove_statistical_outlier(nb_neighbors=neighbors,std_ratio=std)
    #display_inlier_outlier(ply, ind)

    ply = ply.select_by_index(ind)
    post_filt_size = len(ply.points)

    print("Removed {:0.2f}% of points [{} -> {}]...".format(100*(pre_filt_size-post_filt_size)/pre_filt_size, pre_filt_size, post_filt_size))
    return ply

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def main():
    if(len(sys.argv) != 4):
        print("Error: usage python3 {} <data-path> <filter> <merged-file-path>".format(sys.argv[0]))
        sys.exit()

    data_path = sys.argv[1]
    filt = int(sys.argv[2])
    merged_file_path = sys.argv[3]

    if os.path.exists(merged_file_path):
        os.remove(merged_file_path)


    ply_list = read_points(data_path)
    ply = merge_clouds(ply_list)

    if (filt == 1):
        neighbors = 25
        std = 0.1
        ply = remove_outliers(ply, neighbors, std)
    
    save_ply(ply, merged_file_path)


if __name__=="__main__":
    main()
