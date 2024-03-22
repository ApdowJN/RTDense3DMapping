#!/bin/bash

DATA_PATH=/home/wangweihan/Documents/my_project/underwater_project/dataset/Stavronikita/

IN_FILE=${DATA_PATH}colmap_stereo_scaled_traj_left.txt
OUT_FILE=${DATA_PATH}filename_mapping.csv

awk '{split($1,ts,"."); print $1 "," ts[1] ts[2] ".png"}' $IN_FILE > $OUT_FILE

