import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys
import os
import argparse
import scipy.io as sio


# argument parsing
parse = argparse.ArgumentParser(description="Point Cloud Comparison Tool.")

parse.add_argument("-r1", "--src_per_frame_ply", default="../src.ply", type=str, help="Path to source point cloud file.")
parse.add_argument("-r2", "--src_per_seq_ply", default="../src.ply", type=str, help="Path to source point cloud file.")
parse.add_argument("-t", "--tgt_ply", default="../tgt.ply", type=str, help="Path to target point cloud file.")
parse.add_argument("-o", "--output_path", default="./evaluation", type=str, help="Output path where all metrics and results will be stored.")
parse.add_argument("-v", "--voxel_size", default=0.2, type=float, help="Voxel size used for consistent downsampling.")
parse.add_argument("-x", "--max_dist", default=0.4, type=float, help="Max distance threshold for point matching.")
parse.add_argument("-k", "--mask_th", default=20, type=float, help="Masking threshold to remove outliers from comparison.")

ARGS = parse.parse_args()


def correct_round(n):
    return np.round(n+0.5)

def read_point_cloud(ply_path, size=0.1):
    if(ply_path[-3:] != "ply"):
        print("{} is not a '.ply' file.".format(ply_path))
        sys.exit()

    ply = o3d.io.read_point_cloud(ply_path)
    ply = ply.voxel_down_sample(voxel_size=size)

    return ply

def build_src_points_filter(ply, min_bound, res, mask):
    points = np.asarray(ply.points).transpose()
    shape = points.shape
    mask_shape = mask.shape
    filt = np.zeros(shape[1])

    min_bound = min_bound.reshape(3,1)
    min_bound = np.tile(min_bound, (1,shape[1]))

    qv = points
    qv = (points - min_bound) / res
    qv = correct_round(qv).astype(int)

    # get all valid points
    in_bounds = np.asarray(np.where( ((qv[0,:]>=0) & (qv[0,:] < mask_shape[0]) & (qv[1,:]>=0) & (qv[1,:] < mask_shape[1]) & (qv[2,:]>=0) & (qv[2,:] < mask_shape[2])))).squeeze(0)
    valid_points = qv[:,in_bounds]

    # convert 3D coords ([x,y,z]) to appropriate flattened coordinate ((x*mask_shape[1]*mask_shape[2]) + (y*mask_shape[2]) + z )
    mask_inds = np.ravel_multi_index(valid_points, dims=mask.shape, order='C')

    # further trim down valid points by mask value (keep point if mask is True)
    mask = mask.flatten()
    valid_mask_points = np.asarray(np.where(mask[mask_inds] == True)).squeeze(0)

    # add 1 to indices where we want to keep points
    filt[in_bounds[valid_mask_points]] = 1

    return filt

def build_tgt_points_filter(ply, P):
    points = np.asarray(ply.points).transpose()
    shape = points.shape

    # compute iner-product between points and the defined plane
    Pt = P.transpose()

    points = np.concatenate((points, np.ones((1,shape[1]))), axis=0)
    plane_prod = (Pt @ points).squeeze(0)

    # get all valid points
    filt = np.asarray(np.where((plane_prod > 0), 1, 0))

    return filt


def compare_per_frame_per_seq_rectification(src_ply_per_frame, src_ply_per_seq, tgt_ply, mask_th, max_dist):
    # 1. compute bi-directional distance between point clouds
    # 1.1 point cloud from per_frame rectification
    dists_src_per_frame = np.asarray(src_ply_per_frame.compute_point_cloud_distance(tgt_ply))
    valid_dists = np.where(dists_src_per_frame <= mask_th)[0]
    dists_src_per_frame = dists_src_per_frame[valid_dists]

    dists_tgt_per_frame = np.asarray(tgt_ply.compute_point_cloud_distance(src_ply_per_frame))
    valid_dists = np.where(dists_tgt_per_frame <= mask_th)[0]
    dists_tgt_per_frame = dists_tgt_per_frame[valid_dists]

    # 1.2 point cloud from per_seq rectification
    dists_src_per_seq = np.asarray(src_ply_per_seq.compute_point_cloud_distance(tgt_ply))
    valid_dists = np.where(dists_src_per_seq <= mask_th)[0]
    dists_src_per_seq = dists_src_per_seq[valid_dists]

    dists_tgt_per_seq = np.asarray(tgt_ply.compute_point_cloud_distance(src_ply_per_seq))
    valid_dists = np.where(dists_tgt_per_seq <= mask_th)[0]
    dists_tgt_per_seq = dists_tgt_per_seq[valid_dists]

    # 2. compute accuracy and competeness
    # 2.1.1 following are from per_frame rectification
    acc1 = np.mean(dists_src_per_frame)
    comp1 = np.mean(dists_tgt_per_frame)

    # 2.1.2 measure incremental precision and recall values with thesholds from (0, 10*max_dist)
    th_vals = np.linspace(0, 3 * max_dist, num=50)
    prec_vals1 = [(len(np.where(dists_src_per_frame <= th)[0]) / len(dists_src_per_frame)) for th in th_vals]
    rec_vals1 = [(len(np.where(dists_tgt_per_frame <= th)[0]) / len(dists_tgt_per_frame)) for th in th_vals]

    # 2.2.1 following are from per_sequence rectification
    acc2 = np.mean(dists_src_per_seq)
    comp2 = np.mean(dists_tgt_per_seq)

    # 2.2.2 measure incremental precision and recall values with thesholds from (0, 10*max_dist)
    prec_vals2 = [(len(np.where(dists_src_per_seq <= th)[0]) / len(dists_src_per_seq)) for th in th_vals]
    rec_vals2 = [(len(np.where(dists_tgt_per_seq <= th)[0]) / len(dists_tgt_per_seq)) for th in th_vals]


    # compute precision and recall for given distance threshold
    prec1 = len(np.where(dists_src_per_frame <= max_dist)[0]) / len(dists_src_per_frame)
    rec1 = len(np.where(dists_tgt_per_frame <= max_dist)[0]) / len(dists_tgt_per_frame)

    prec2 = len(np.where(dists_src_per_seq <= max_dist)[0]) / len(dists_src_per_seq)
    rec2 = len(np.where(dists_tgt_per_seq <= max_dist)[0]) / len(dists_tgt_per_seq)

    src_per_frame_size = len(src_ply_per_frame.points)
    src_per_seq_size = len(src_ply_per_seq.points)
    tgt_size = len(tgt_ply.points)

    return (acc1, comp1), (acc2, comp2), (prec1, rec1), (prec2, rec2), (th_vals, prec_vals1, rec_vals1), (th_vals, prec_vals2, rec_vals2), (src_per_frame_size, src_per_seq_size, tgt_size)


def save_ply(file_path, ply):
    o3d.io.write_point_cloud(file_path, ply)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def main():
    ##### Initialization #####
    # set parameters
    src_per_frame_path = ARGS.src_per_frame_ply
    src_per_seq_path = ARGS.src_per_seq_ply
    tgt_path = ARGS.tgt_ply
    output_path = ARGS.output_path
    voxel_size = ARGS.voxel_size
    max_dist = ARGS.max_dist
    mask_th = ARGS.mask_th

    # create output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    ##### Load in point clouds #####
    print("Loading point clouds...")
    src_per_frame_ply = read_point_cloud(src_per_frame_path, voxel_size)
    src_per_seq_ply = read_point_cloud(src_per_seq_path, voxel_size)
    tgt_ply = read_point_cloud(tgt_path, voxel_size)


    ##### Compute metrics between point clouds #####
    print("Computing metrics between point clouds...")
    (acc1, comp1), (acc2, comp2), (prec1, rec1), (prec2, rec2), (th_vals, prec_vals1, rec_vals1), (th_vals, prec_vals2, rec_vals2), (src_per_frame_size, src_per_seq_size, tgt_size) \
            = compare_per_frame_per_seq_rectification(src_per_frame_ply,src_per_seq_ply, tgt_ply, mask_th, max_dist)


    # create plots for incremental threshold values
    plot_filename = os.path.join(output_path, "metrics.png")
    plt.plot(th_vals, prec_vals1, th_vals, rec_vals1, th_vals, prec_vals2, th_vals, rec_vals2)
    plt.title("Precision and Recall (t={}m)".format(max_dist))
    plt.xlabel("threshold")
    plt.vlines(max_dist, 0, 1, linestyles='dashed', label='t')
    plt.legend(("precision (per_frame)", "recall (per_frame)", "precision (per_seq)", "recall (per_seq)"))
    plt.grid()
    plt.savefig(plot_filename)

    # write all metrics to the evaluation file
    # python read/write
    stats_file = os.path.join(output_path, "evaluation_metrics.txt")
    with open(stats_file, 'w') as f:
        f.write("Voxel_size: {:0.3f}m | Distance threshold: {:0.3f}m\n".format(voxel_size, max_dist))
        f.write("Source point cloud (per_frame rectification) size: {}\n".format(src_per_frame_size))
        f.write("Source point cloud (per_sequence rectification) size: {}\n".format(src_per_seq_size))
        f.write("Target point cloud size: {}\n".format(tgt_size))
        f.write("Accuracy (per_frame rectification): {:0.3f}m\n".format(acc1))
        f.write("Accuracy (per_sequence rectification): {:0.3f}m\n".format(acc2))
        f.write("Completness (per_frame rectification): {:0.3f}m\n".format(comp1))
        f.write("Completness (per_sequence rectification): {:0.3f}m\n".format(comp2))
        f.write("Precision (per_frame rectification): {:0.3f}\n".format(prec1))
        f.write("Precision (per_sequence rectification): {:0.3f}\n".format(prec2))
        f.write("Recall (per_frame rectification): {:0.3f}\n".format(rec1))
        f.write("Recall (per_sequence rectification): {:0.3f}\n".format(rec2))

if __name__=="__main__":
    main()


