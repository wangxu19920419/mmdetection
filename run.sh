#!/usr/bin/env bash
# pip3 install numpy torchvision_nightly
# pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
# pip3 install torch torchvision
# cd /opt/tiger/
# cp -r /mnt/cephfs_wj/cv/luyifan/cuda-10.0 ./
# export PATH=/opt/tiger/cuda-10.0/bin:$PATH
# export LD_LIBRARY_PATH=/opt/tiger/cuda-10.0/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=/opt/tiger/cuda-10.0/lib64:$LIBRARY_PATH
# export CUDA_HOME='/opt/tiger/cuda-10.0/'
# nvcc --version
# cd -
# pip3 install /mnt/cephfs_wj/cv/luyifan/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
# pip3 install http://pypi.byted.org/root/pypi/+f/ba0/a5e7fa1646d4e/torchvision-0.2.1-py2.py3-none-any.whl
# PATH=/usr/local/cuda-9.0/bin/:$PATH
# PATH=/usr/local/cuda-10.0/bin/:$PATH

### change pip source image
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p /root/.torch/models
cp /mnt/cephfs_wj/cv/wangxu.ailab/pytorch/bytedetection/pretrain_models/* /mnt/cephfs_wj/cv/wangxu.ailab/ideas_experiments/mmdetection/pretrain_models/
cp /mnt/cephfs_wj/cv/wangxu.ailab/ideas_experiments/mmdetection/pretrain_models/* /root/.torch/models/
cp -r /mnt/cephfs_wj/cv/wangxu.ailab/ideas_experiments/mmdetection /opt/tiger/run_arnold
cp -r /mnt/cephfs_wj/cv/wangxu.ailab/pytorch/apex /opt/tiger/run_arnold/mmdetection/
cd /opt/tiger/run_arnold/mmdetection/apex/
echo 'Changing Directory!!!!!!!!!!!!!!'
pwd
python3 setup.py install --cpp_ext --cuda_ext
# python3 setup.py install

cp -r /mnt/cephfs_wj/cv/wangxu.ailab/pytorch/mmcv /opt/tiger/run_arnold/mmdetection/mmcv
cd /opt/tiger/run_arnold/mmdetection/mmcv/
echo 'Changing Directory!!!!!!!!!!!!!!'
pwd
pip3 install -e .

#rsync -a --exclude work_dirs /mnt/cephfs_wj/cv/wangxu.ailab/pytorch/bytedetection /opt/tiger/bytedetection/
cd /opt/tiger/run_arnold/mmdetection
echo 'Changing Directory!!!!!!!!!!!!!!'
pwd
ls
PYTHON=python3 ./compile.sh
python3 setup.py install
#cp /opt/tiger/bytedetection/mmdet/models/backbones/resnet.py /opt/tiger/bytedetection/build/lib/mmdet/models/backbones/resnet.py

####### TRAINING
# ./tools/dist_train.sh configs/faster_rcnn_r50_fpn_1x_arnold.py 8
# ./tools/dist_train.sh configs/cascade_rcnn_r50_fpn_2x_arnold.py 4
# ./tools/dist_train.sh configs/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_2x_arnold.py 6
# ./tools/dist_train.sh configs/dcn/cascade_rcnn_dconv_c3-c5_r101_fpn_arnold.py 7
##0315 11:34
# ./tools/dist_train.sh configs/cascade_rcnn_r50_fpn_2x_arnold.py 4
# ###0315 11:37
# ./tools/dist_train.sh configs/cascade_rcnn_r50_fpn_2x_arnold.py 8

###0317 22.25
# ./tools/dist_train.sh configs/dcn/cascade_rcnn_dconv_c3-c5_r101_fpn_apex_arnold.py 8
###0318 22:43
#export WORLD_SIZE=2
#export RANK=$ARNOLD_ID
#export MASTER_ADDR=`echo $ARNOLD_WORKER_HOSTS | awk -F: '{ print $1 }'`
#export MASTER_PORT=`echo $ARNOLD_WORKER_HOSTS | awk -F: '{ print $2 }'`
# ./tools/dist_train.sh configs/cascade_rcnn_se154_fpn_2x_arnold.py 8

####VIC
# hosts=$ARNOLD_WORKER_HOSTS
# IFS=','
# HOST_CONFIGS=($hosts)
# unset IFS
# IFS=':'
# WORKER0=(${HOST_CONFIGS[0]})
# WORKER0_IP=${WORKER0[0]}
# WORKER0_PORT=${WORKER0[1]}

###VIC

### 20190320
# python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
#   --nnodes=$ARNOLD_NUM \
#   --node_rank=$ARNOLD_ID \
#   --master_addr=$WORKER0_IP \
#   --master_port=$WORKER0_PORT \
#   tools/train.py configs/cascade_rcnn_se154_fpn_2x_arnold.py \
#   --launcher pytorch

### 20190321
# python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
#  --nnodes=$ARNOLD_NUM \
#  --node_rank=$ARNOLD_ID \
#  --master_addr=$WORKER0_IP \
#  --master_port=$WORKER0_PORT \
#  tools/train.py configs/dcn/cascade_rcnn_dconv_c3-c5_se154_fpn_apex_arnold.py \
#   --launcher pytorch

### 20190323
# python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
#  --nnodes=$ARNOLD_NUM \
#  --node_rank=$ARNOLD_ID \
#  --master_addr=$WORKER0_IP \
#  --master_port=$WORKER0_PORT \
#  tools/train.py configs/dcn/cascade_rcnn_dconv2_c3-c5_se154_fpn_ohem_apex_arnold.py \
#  --launcher pytorch

### 20190326
# python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
#  --nnodes=$ARNOLD_NUM \
#  --node_rank=$ARNOLD_ID \
#  --master_addr=$WORKER0_IP \
#  --master_port=$WORKER0_PORT \
#  tools/train.py configs/dcn/cascade_rcnn_dconv_c5_se154_fpn_apex_arnold.py \
#  --launcher pytorch

### 20190327
###VIC
# python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
#  --nnodes=$ARNOLD_NUM \
#  --node_rank=$ARNOLD_ID \
#  --master_addr=$WORKER0_IP \
#  --master_port=$WORKER0_PORT \
#  tools/train.py configs/dcn/cascade_rcnn_dconv_c45-c5_se154_fpn_apex_arnold.py \
#  --launcher pytorch
./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py 4
#./tools/dist_train.sh configs/dcn/cascade_rcnn_dconv_c45-c5_se154_fpn_apex_arnold.py 1
 ###VIC

####### TESTING
# python3 tools/test.py configs/dcn/cascade_rcnn_dconv_c3-c5_r101_fpn_arnold.py \
#         work_dirs/cascade_rcnn_dconv_c3-c5_r101_fpn_arnold/epoch_15.pth --gpus 6 \
#         --out work_dirs/cascade_rcnn_dconv_c3-c5_r101_fpn_arnold_results_multiscale2.pkl --eval bbox
