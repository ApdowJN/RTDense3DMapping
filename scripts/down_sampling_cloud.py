import numpy as np
import sys
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import argparse

def read_point_cloud(ply_path, size=0.1):
    if(ply_path[-3:] != "ply"):
        print("{} is not a '.ply' file.".format(ply_path))
        sys.exit()

    ply = o3d.io.read_point_cloud(ply_path)
    ply = ply.voxel_down_sample(voxel_size=size)

    return ply

def save_ply(ply, file_path):
    print("Saving downsampled point cloud...")
    o3d.io.write_point_cloud(file_path, ply)

def main():
    if (len(sys.argv) != 4):
        print("Error: usage python3 {} <data-path> <filter> <merged-file-path>".format(sys.argv[0]))
        sys.exit()

    src_path = sys.argv[1]
    voxel_size = float(sys.argv[2])
    downsample_save_path = sys.argv[3]

    ##### Load in point clouds #####
    print("Loading point clouds...")
    src_ply = read_point_cloud(src_path, voxel_size)
    print("Save downsimped point clouds...")
    save_ply(src_ply, downsample_save_path)
if __name__ == "__main__":
    main()
