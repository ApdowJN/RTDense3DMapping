#!/bin/bash

# Data paths
DATA_PATH=/home/wangweihan/Documents/my_project/underwater_project/dataset/Stavronikita/
OUTPUT_PATH=/home/wangweihan/Documents/my_project/underwater_project/dataset/Output/Stavronikita_per_seq/

#DATA_PATH=/media/nate/Data/NSF/Stavronikita/
#OUTPUT_PATH=/media/nate/Data/NSF/Output/Stavronikita/

# Parameter setup
SCENE=STAVRONIKITA
INTRINSICS_FILE=intrinsics.yaml
POSE_FILE=colmap_stereo_scaled_traj_left.txt
FILENAME_MAPPING=filename_mapping.csv
OUTPUT_PCL_FILE=stavronikita.ply

NUM_VIEWS=3
CONF_PRE_TH=0.2
CONF_POST_TH=0.5
SUPPORT_RATIO=0.04
ALIGNMENT_TH=20
NUM_MATCHES=50

NUM_DISP=100
WINDOW_SIZE=3
METHOD=sad

VOXEL_SIZE=0.001 
MAX_DIST=0.10 #10cm

BUILD_DIR=../build/
EXE_NAME=accelerated_depth_fusion #depth_fusion2
EXE_PATH=../bin/${EXE_NAME}



# compilation function
build_exe() {
    if [ ! -d ${1} ]; then
        mkdir -p ${1};
    fi

    cd ${1}
	rm ${EXE_PATH} 2> /dev/null
    cmake -DCMAKE_BUILD_TYPE=Debug .. > /dev/null
	make -j $(nproc)
}

create_output_dir() {
	mkdir "${OUTPUT_PATH}";
	mkdir "${OUTPUT_PATH}/conf_maps";
	mkdir "${OUTPUT_PATH}/disp_maps";
	mkdir "${OUTPUT_PATH}/depth_maps";
	mkdir "${OUTPUT_PATH}/fused_maps";
	mkdir "${OUTPUT_PATH}/fused_maps/display";
	mkdir "${OUTPUT_PATH}/point_clouds";
	mkdir "${OUTPUT_PATH}/rectified_images";
	mkdir "${OUTPUT_PATH}/rectified_images/left";
	mkdir "${OUTPUT_PATH}/rectified_images/right";
	mkdir "${OUTPUT_PATH}/feature_matching";
}

display_params() {
	echo "${DATE}"
	echo "Intrinsics file set to '${INTRINSICS_FILE}'..."
	echo "Poses file set to '${POSE_FILE}'..."
	echo "Filename mapping file set to '${FILENAME_MAPPING}'..."
	echo "Number of views for fusion set to '${NUM_VIEWS}'..."
	echo "Fusion Confidence pre-threshold set to '${CONF_PRE_TH}'..."
	echo "Fusion confidence post-threshold set to '${CONF_POST_TH}'..."
	echo "Fusion support ratio set to '${SUPPORT_RATIO}'..."
	echo "Stereo pair alignment rejection threshold set to '${ALIGNMENT_TH}'..."
	echo "Number of features to match set to '${NUM_MATCHES}'..."
	echo "Stereo number of disparities set to '${NUM_DISP}'..."
	echo "Stereo window size set to '${WINDOW_SIZE}'..."
	echo "Stereo method set to '${METHOD}'..."
	echo "Voxel size set to '${VOXEL_SIZE}'..."
	echo "Max distance threshold set to '${MAX_DIST}'..."
	echo -e "\n"
}



# compile executable
build_exe "${BUILD_DIR}" &
wait


# clean old output data
if [ -d ${OUTPUT_PATH} ]; then
    rm -rf ${OUTPUT_PATH};
fi
create_output_dir &
wait


# logging
DATE=$(date +"%F-%T")
LOG_PATH=${OUTPUT_PATH}log/
if [ ! -d ${LOG_PATH} ]; then
    mkdir -p ${LOG_PATH};
    touch ${LOG_PATH}/log.txt;
fi


# display parameters
display_params | tee -a ${LOG_PATH}log.txt &
wait

# run pipeline
$EXE_PATH \
		$SCENE \
		${DATA_PATH}$INTRINSICS_FILE \
		${DATA_PATH}${POSE_FILE} \
		${DATA_PATH} \
		${OUTPUT_PATH} \
		$NUM_VIEWS \
		$CONF_PRE_TH \
		$CONF_POST_TH \
		$SUPPORT_RATIO \
		${DATA_PATH}${FILENAME_MAPPING} \
		${OUTPUT_PATH}${OUTPUT_PCL_FILE} \
		${ALIGNMENT_TH} \
		${NUM_DISP} \
		${WINDOW_SIZE} \
		${NUM_MATCHES} \
		${METHOD} | tee -a ${LOG_PATH}log.txt

# plot alignment info
# python plot_logs.py ${LOG_PATH}avg_offsets.csv ${LOG_PATH}std_dev.csv ${LOG_PATH}xy_offsets.csv ${LOG_PATH}radial_offsets.csv ${LOG_PATH}

# # fuse point clouds (filtered(1) and unfiltered(0))
# #python merge_point_clouds.py ${OUTPUT_PATH}point_clouds/ 0 ${OUTPUT_PATH}stavronikita_unfiltered.ply
# python merge_point_clouds.py ${OUTPUT_PATH}point_clouds/ 0 ${OUTPUT_PATH}stavronikita_filtered.ply

# compare output to colmap
python compare_clouds.py -r ${OUTPUT_PATH}${OUTPUT_PCL_FILE} -t ${DATA_PATH}colmap.ply -o ${OUTPUT_PATH}evaluation/ --voxel_size ${VOXEL_SIZE} --max_dist $MAX_DIST
# compare filtered depth maps to colmap
# python compare_depth_maps.py -n ${SCENE} -r1 ${OUTPUT_PATH}point_clouds/ -r2 ${DATA_PATH}${POSE_FILE} -r3 ${OUTPUT_PATH}fused_maps/ -t ${DATA_PATH}colmap_depth_maps/left/ -o ${OUTPUT_PATH}evaluation/
