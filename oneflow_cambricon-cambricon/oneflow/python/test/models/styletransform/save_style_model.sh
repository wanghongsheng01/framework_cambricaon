#!/bin/bash
set -ex

# download model parameters for first-time 
# wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/styletranform.tar.gz
# tar zxvf styletranform.tar.gz

base_dir=`dirname $0`
BACKEND="${1:-gpu}"
SERVING_MODEL_NAME="style_transform_models_${BACKEND}"

python3 $base_dir/save_style_model.py \
    --backend $BACKEND \
    --model_dir stylenet_nhwc \
    --save_dir $SERVING_MODEL_NAME \
    --model_version 1 \
    --image_width 640 \
    --image_height 640 \
    --force_save
