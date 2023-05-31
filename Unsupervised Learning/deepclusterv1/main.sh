# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# RESUME="deepclusterv1/exp/checkpoint.pth.tar" --resume ${RESUME}
#
#!/bin/bash
#DIR="/home/bigdatalab/Github/Data-Lab/Projects/Coin_collection_3/Die_comparison/data/imgs/Class_VI/Filter4/"
DIR="/home/bigdatalab/Projects/D4N4/CN_mints_types/daten/deep_new/"
ARCH="vgg16"
LR=0.05
WD=-5
K=15
WORKERS=8
EXP="deepclusterv1/export/d4n4_new_classes/"
PYTHON="/home/bigdatalab/anaconda3/envs/pytorch/bin/python"
EPOCHS=350
BATCH=64
ITER=1
#CHANGE_CLUSTER=10
#CHANGE_ALGORITHM="hierarchical_clustering" # kmeans_modified
#RESUME="deepclusterv1/export/MS1_obverse_filter/checkpoint.pth.tar" 
mkdir ${EXP}
CUDA_VISIBLE_DEVICES=0 ${PYTHON} deepclusterv1/main.py ${DIR} --exp ${EXP} --arch ${ARCH} --reassign ${ITER} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} --epochs ${EPOCHS} --batch ${BATCH} \
  #--change_algorithm ${CHANGE_ALGORITHM} --change_cluster ${CHANGE_CLUSTER} ##--resume ${RESUME}
