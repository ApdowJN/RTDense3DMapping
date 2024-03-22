#!/usr/bin/env python

import argparse
import numpy as np
import os
import re
import pylab as plt
from tqdm import tqdm
import open3d as o3d
import quaternion
def read_colmap_depth_maps(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def read_pipeline_depth_maps(path):
    file = open(path, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def parse_args1():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--src_folder",
                        help="path to pipeline's depth maps", type=str, required=True)
    parser.add_argument("-t", "--tgt_folder",
                        help="path to COLMAP's depth maps", type=str, required=True)
    parser.add_argument("-o", "--output_path", default="./evaluation", type=str,
                       help="Output path where all metrics and results will be stored.")
    parser.add_argument("-p", "--photo_metric", action="store_true",
                        help="if set, read photometric depth maps. else by defualt geometric")
    parser.add_argument("--min_depth_percentile",
                        help="minimum visualization depth percentile",
                        type=float, default=5)
    parser.add_argument("--max_depth_percentile",
                        help="maximum visualization depth percentile",
                        type=float, default=95)

    args = parser.parse_args()
    return args

def compare_fused_depth_maps():
    args = parse_args1()

    if args.min_depth_percentile > args.max_depth_percentile:
        raise ValueError("min_depth_percentile should be less than or equal "
                         "to the max_depth_perceintile.")

    # Read depth/normal maps from folder
    if not os.path.exists(args.tgt_folder):
        raise FileNotFoundError("Folder not found: {}".format(args.depth_map))

    # regex_exp = re.compile(r'photometric') if args.photo_metric else re.compile(r'geometric')
    # colmap_depth_map_paths = [os.path.join(args.tgt_folder, f) for f in os.listdir(args.tgt_folder) if regex_exp.search(f)]
    # colmap_depth_map_paths.sort()
    # Get the name of depth map in COLMAP eg:1638570925380233100
    # patten = args.tgt_folder + '(.+?).png.geometric.bin'
    # whole_colmap_depth_map_names_2d = [re.findall(patten, colmap_depth_map_path) for colmap_depth_map_path in colmap_depth_map_paths]
    # whole_colmap_depth_map_names = [i for iterm in whole_colmap_depth_map_names_2d for i in iterm] # eg: whole_colmap_depth_map_names = [1638570925380233100, 1638570925380233101]
    # whole_colmap_depth_map_names.sort()

    regex_exp = re.compile(r'pfm')
    pipeline_depth_map_paths = [os.path.join(args.src_folder, f) for f in os.listdir(args.src_folder) if regex_exp.search(f)]
    pipeline_depth_map_paths.sort()

    patten = args.src_folder + '(.+?).pfm'
    pipeline_depth_map_names_2d = [re.findall(patten, depth_map_path) for depth_map_path in pipeline_depth_map_paths]
    pipeline_depth_map_names = [i for iterm in pipeline_depth_map_names_2d for i in iterm]
    pipeline_depth_map_names.sort()
    total_mae = 0.0
    pipeline_depth_map_names = pipeline_depth_map_names[1:9]
    stats_file = os.path.join(args.output_path, "evaluation_raw_depth_maps.txt")
    with open(stats_file, 'w') as f:
        for i in tqdm(range(len(pipeline_depth_map_names)), desc="Loading and comparing depth maps", unit="depth maps"):
            colmap_depth_map_path = args.tgt_folder + pipeline_depth_map_names[i] +'.png.geometric.bin'
            pipline_depth_map_path = pipeline_depth_map_paths[i]

            colmap_depth_map = read_colmap_depth_maps(colmap_depth_map_path)
            pipeline_depth_map = read_pipeline_depth_maps(pipline_depth_map_path)

            both_non_zeros = (pipeline_depth_map != 0) & (colmap_depth_map != 0)
            both_zeros = (pipeline_depth_map == 0) & (colmap_depth_map == 0)

            mae = np.mean(np.absolute(pipeline_depth_map[both_non_zeros] - colmap_depth_map[both_non_zeros]))
            total_mae += mae
            valid_pipeline_depths = np.count_nonzero(pipeline_depth_map)
            valid_colmap_depths = np.count_nonzero(colmap_depth_map)

            # write all metrics to the evaluation file
            # python read/write
            f.write("No.{0} Image name: {1}\n".format(i, pipeline_depth_map_names[i]))
            f.write("MAE: {}\n".format(mae))
            f.write("Number of pixels with depth (non zeros) by Pipeline: {}\n".format(valid_pipeline_depths))
            f.write("Number of pixels with depth (non zeros) by COLMAP: {}\n".format(valid_colmap_depths))
            f.write("Average depth in Pipeline's depth map: {}\n".format(np.mean(pipeline_depth_map[pipeline_depth_map!=0])))
            f.write("Average depth in COLMAP's depth map: {}\n".format(np.mean(colmap_depth_map[colmap_depth_map!=0])))
            f.write("Number of pixels on which both Pipeline and COLMAP are zeros: {}\n".format(np.sum(both_zeros)))
            f.write("=====================================================================================================\n")
        f.write("MAE in all depth maps: {}\n".format(total_mae/len(pipeline_depth_map_names)))
    f.close()

def parse_args2():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--dataset_name",
                        help="dataset name", type=str, required=True)
    parser.add_argument("-r1", "--src_points_folder",
                        help="path to pipeline's point clouds", type=str, required=True)
    parser.add_argument("-r2", "--src_poses_path",
                        help="path to pipeline's poses", type=str, required=True)
    parser.add_argument("-r3", "--src_depth_folder",
                        help="path to pipeline's depth maps", type=str, required=True)
    parser.add_argument("-t", "--tgt_folder",
                        help="path to COLMAP's depth maps", type=str, required=True)
    parser.add_argument("-o", "--output_path", default="./evaluation", type=str,
                       help="Output path where all metrics and results will be stored.")
    parser.add_argument("-p", "--photo_metric", action="store_true",
                        help="if set, read photometric depth maps. else by defualt geometric")

    args = parser.parse_args()
    return args

def load_poses(path):
    f = open(path, "r")
    lines = f.readlines()
    img_name_poses_map = {}

    for line in lines:
        tokens = line.split(" ")
        timestamp = tokens[0]
        timestamp_tokens = timestamp.split(".")
        img_name = timestamp_tokens[0]+timestamp_tokens[1]
        twc = np.array([np.float64(tokens[1]), np.float64(tokens[2]), np.float64(tokens[3])])
        qwc = np.quaternion(np.float64(tokens[7]), np.float64(tokens[4]), np.float64(tokens[5]), np.float64(tokens[6]))
        Rwc = quaternion.as_rotation_matrix(qwc)
        Rcw = np.transpose(Rwc)
        tcw = -Rcw.dot(twc)
        img_name_poses_map[img_name] = (Rcw, tcw)
    f.close()
    return img_name_poses_map

def read_points_cloud(ply_path):
    model = o3d.io.read_point_cloud(ply_path)
    return np.asarray(model.points)

def read_points_clouds(data_path):
    points = []
    file_list = os.listdir(data_path)
    file_list.sort()

    for pc in tqdm(file_list, desc="Loading Point Clouds", unit="point-clouds"):
        if(pc[-3:] != "ply"):
            continue
        ply_path = os.path.join(data_path, pc)
        ply = o3d.io.read_point_cloud(ply_path)
        points.append((np.asarray(ply.points)))

    return points

def compare_filtered_fused_depth_maps():
    args = parse_args2()
    dataset = args.dataset_name
    print("Compare filtered fused depth maps with COLMAP's in {0} Dataset".format(dataset))
    # Read depth/normal maps from folder
    if not os.path.exists(args.tgt_folder):
        raise FileNotFoundError("Folder not found: {}".format(args.depth_map))
    if dataset == 'FLORIDA':
        K = np.array([[595.58148, 0, 380.47882], [0, 593.81262, 302.91428], [0, 0, 1]])
    elif dataset == 'MEXICO':
        K = np.array([[600.8175, 0, 383.27002], [0, 600.82904, 287.87112], [0, 0, 1]])
    elif dataset == 'STAVRONIKITA':
        K = np.array([[1289.2439, 0, 279.92609], [0, 1290.3586, 313.35413], [0, 0, 1]])
    elif dataset == 'REEF':
        K = np.array([[1289.2439, 0, 279.92609], [0, 1290.9395, 313.69763], [0, 0, 1]])
    elif dataset == 'PAMIR':
        K = np.array([[1289.2439, 0, 279.92609], [0, 1290.9395, 313.69763], [0, 0, 1]])
    else:
        print("Input dataset is not supported now")
        return

    img_SE3_map = load_poses(args.src_poses_path)

    regex_exp = re.compile(r'pfm')
    pipeline_depth_map_paths = [os.path.join(args.src_depth_folder, f) for f in os.listdir(args.src_depth_folder) if
                                regex_exp.search(f)]
    pipeline_depth_map_paths.sort()

    patten = args.src_depth_folder + '(.+?).pfm'
    pipeline_depth_map_names_2d = [re.findall(patten, depth_map_path) for depth_map_path in pipeline_depth_map_paths]
    pipeline_depth_map_names = [i for iterm in pipeline_depth_map_names_2d for i in iterm]
    pipeline_depth_map_names.sort()

    stats_file = os.path.join(args.output_path, "evaluation_depth_maps.txt")
    global_min_mae = 100.0
    global_max_mae = 0.0
    maes = []
    medians = []
    num_pixels = []
    total_both_no_zeros_pixels = 0

    for i in tqdm(range(len(pipeline_depth_map_names)), desc="Loading and comparing depth maps", unit="depth maps"):
        colmap_depth_map_path = args.tgt_folder + pipeline_depth_map_names[i] +'.png.geometric.bin'
        pipline_depth_map_path = pipeline_depth_map_paths[i]
        # depth maps
        colmap_depth_map = read_colmap_depth_maps(colmap_depth_map_path)
        pipeline_fused_depth_map = read_pipeline_depth_maps(pipline_depth_map_path)
        pipeline_filtered_depth_map = np.zeros((600, 800))
        # pipeline's 3D model
        model_path = args.src_points_folder + pipeline_depth_map_names[i] +".ply"
        model_points = read_points_cloud(model_path)
        if len(model_points) == 0:
            continue
        for point in model_points:
            Rcw = img_SE3_map[pipeline_depth_map_names[i]][0]
            tcw = img_SE3_map[pipeline_depth_map_names[i]][1]
            pc = Rcw.dot(point) + tcw
            pixels = K.dot(pc)
            depth = pixels[2]
            u = pixels[0]/pixels[2]
            v = pixels[1]/pixels[2]
            if u < 0 or u >= 800 or v < 0 or v >= 600:
                continue
            pipeline_filtered_depth_map[round(v), round(u)] = depth

        both_non_zeros = (pipeline_filtered_depth_map != 0) & (colmap_depth_map != 0)
        both_zeros = (pipeline_filtered_depth_map == 0) & (colmap_depth_map == 0)
        if(np.sum(both_non_zeros) == 0):
            continue
        median = np.median(np.absolute(pipeline_filtered_depth_map[both_non_zeros] - colmap_depth_map[both_non_zeros]))
        medians.append(median)
        mae = np.mean(np.absolute(pipeline_filtered_depth_map[both_non_zeros] - colmap_depth_map[both_non_zeros]))
        maes.append(mae)

        global_min_mae = min(mae, global_min_mae)
        global_max_mae = max(mae, global_max_mae)

        N_both_non_zeros = np.sum(both_non_zeros)
        num_pixels.append(N_both_non_zeros)
        total_both_no_zeros_pixels+=N_both_non_zeros

    weighted_average_mae = 0.0
    with open(stats_file, 'w') as f:
        for i in range(len(maes)):
            f.write("No.{0} \n".format(i))
            f.write("MAE: {0} over {1} pixels \n".format(maes[i], num_pixels[i]))
            f.write("Median of error: {0} over {1} pixels \n".format(medians[i], num_pixels[i]))
            weighted_average_mae += (maes[i]*(num_pixels[i]/total_both_no_zeros_pixels))
            f.write(
                "=====================================================================================================\n")
        f.write("Minimum MAE: {0}, Median Medians: {1}, weighted average of MAE {2}, Maximum MAE: {3}\n".format(global_min_mae, np.median(medians), weighted_average_mae,global_max_mae))
    f.close()


if __name__ == "__main__":
    # comparision between fused depth maps and COLMAP's depth maps
    compare_fused_depth_maps()

    # # comparision between filtered fused depth maps (convert from point clouds) and COLMAP's depth maps
    # compare_filtered_fused_depth_maps()
