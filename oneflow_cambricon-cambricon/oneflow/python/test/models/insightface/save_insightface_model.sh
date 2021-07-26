#!/bin/bash
set -ex

# download model parameters for first-time 
# wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/insightface.tar.gz
# tar zxvf insightface.tar.gz

base_dir=`dirname $0`

python3 $base_dir/save_insightface_model.py \
    --model_dir insightface \
    --save_dir insightface_models \
    --model_version 1 \
    --image_width 112 \
    --image_height 112 \
    --force_save
