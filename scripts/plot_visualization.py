#!/usr/bin/env python

import argparse
import numpy as np
import os
import re
import cv2
from tqdm import tqdm

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--src_folder",
                        help="path to pipeline's depth maps", type=str, required=True)
    parser.add_argument("-t", "--tgt_folder",
                        help="path to COLMAP's depth maps", type=str, required=True)
    parser.add_argument("-o", "--output_path", default="./evaluation", type=str,
                       help="Output path where all metrics and results will be stored.")
    parser.add_argument("--min_depth",
                        help="minimum visualization depth",
                        type=float, default=0)
    parser.add_argument("--max_depth",
                        help="maximum visualization depth",
                        type=float, default=5)

    args = parser.parse_args()
    return args

def plot_pfm_vis_png():
    args = parse_args()

    regex_exp = re.compile(r'pfm')
    pipeline_depth_map_paths = [os.path.join(args.src_folder, f) for f in os.listdir(args.src_folder) if regex_exp.search(f)]
    pipeline_depth_map_paths.sort()

    patten = args.src_folder + '(.+?).pfm'
    pipeline_depth_map_names_2d = [re.findall(patten, depth_map_path) for depth_map_path in pipeline_depth_map_paths]
    pipeline_depth_map_names = [i for iterm in pipeline_depth_map_names_2d for i in iterm]
    pipeline_depth_map_names.sort()
    # pipeline_depth_map_names = pipeline_depth_map_names[1:9]


    for i in tqdm(range(len(pipeline_depth_map_names)), desc="Loading and visualize depth maps", unit="depth maps"):
        pipline_depth_map_path = pipeline_depth_map_paths[i]
        pipeline_depth_map = read_pipeline_depth_maps(pipline_depth_map_path)

        pipeline_depth_map[pipeline_depth_map < args.min_depth] = 0
        pipeline_depth_map[pipeline_depth_map >= args.max_depth] = args.max_depth

        vis_img = cv2.normalize(pipeline_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Saving the image
        save_path = args.output_path + pipeline_depth_map_names[i] +".png"
        cv2.imwrite(save_path, vis_img)

def plot_colmap_vis_png():
    args = parse_args()
    regex_exp = re.compile(r'pfm')
    pipeline_depth_map_paths = [os.path.join(args.src_folder, f) for f in os.listdir(args.src_folder) if
                                regex_exp.search(f)]
    pipeline_depth_map_paths.sort()

    patten = args.src_folder + '(.+?).pfm'
    pipeline_depth_map_names_2d = [re.findall(patten, depth_map_path) for depth_map_path in pipeline_depth_map_paths]
    pipeline_depth_map_names = [i for iterm in pipeline_depth_map_names_2d for i in iterm]
    pipeline_depth_map_names.sort()
    # pipeline_depth_map_names = pipeline_depth_map_names[1:9]

    for i in tqdm(range(len(pipeline_depth_map_names)), desc="Loading and visualize colmap's depth maps", unit="depth maps"):
        colmap_depth_map_path = args.tgt_folder + pipeline_depth_map_names[i] + '.png.geometric.bin'
        # depth maps
        colmap_depth_map = read_colmap_depth_maps(colmap_depth_map_path)
        colmap_depth_map[colmap_depth_map < args.min_depth] = 0
        colmap_depth_map[colmap_depth_map >= args.max_depth] = args.max_depth

        vis_img = cv2.normalize(colmap_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Saving the image
        save_path = args.output_path + pipeline_depth_map_names[i] + ".png"
        cv2.imwrite(save_path, vis_img)

if __name__ == "__main__":
    plot_pfm_vis_png()
    #
    # plot_colmap_vis_png()
