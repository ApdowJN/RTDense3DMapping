import numpy as np
import os
import sys
import shutil

def build_mapping(filename_mapping):
    mapping = {}

    with open(filename_mapping,'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.split(',')
            mapping[line[0]] = line[1]

    return mapping
    
def accumulate_keyframes(path, mapping, pose_file):
    # grab list of img names
    timestamps = []
    filenames = []

    with open(pose_file,'r') as pf:
        lines = pf.readlines()
        
        for line in lines:
            line = line.strip().split(' ')
            timestamps.append(line[0])
            fname = mapping[line[0]]
            filenames.append(fname[:-1])

    # create new img list csv
    new_img_list = os.path.join(path, "new_image_list.csv")
    with open(new_img_list,'w') as f:
        for ts,fn in zip(timestamps,filenames):
            f.write("{},{}\n".format(ts,fn))

    # migrate left keyframes
    old_left_path = os.path.join(path, "left")
    new_left_path = os.path.join(path, "left_keyframes")

    # create new dir if it does not exist
    if(not os.path.isdir(new_left_path)):
        os.mkdir(new_left_path)

    # lsit all imgs
    imgs = os.listdir(old_left_path)
    imgs.sort()

    # migrate only keyframe imgs
    for img in imgs:
        if (img in filenames):
            old_file = os.path.join(old_left_path, img)
            new_file = os.path.join(new_left_path, img)
            shutil.move(old_file, new_file)


    # migrate right keyframes
    old_right_path = os.path.join(path, "right")
    new_right_path = os.path.join(path, "right_keyframes")

    # create new dir if it does not exist
    if(not os.path.isdir(new_right_path)):
        os.mkdir(new_right_path)

    # lsit all imgs
    imgs = os.listdir(old_right_path)
    imgs.sort()

    # migrate only keyframe imgs
    for img in imgs:
        if (img in filenames):
            old_file = os.path.join(old_right_path, img)
            new_file = os.path.join(new_right_path, img)
            shutil.move(old_file, new_file)

def main():
    if(len(sys.argv) != 4):
        print("Error: usage python {} <data-path> <filename-mapping> <pose-file>".format(sys.argv[0]))
        sys.exit()

    data_path = sys.argv[1]
    filename_mapping = sys.argv[2]
    pose_file = sys.argv[3]

    mapping = build_mapping(filename_mapping)
    accumulate_keyframes(data_path, mapping, pose_file)

if __name__=="__main__":
    main()
