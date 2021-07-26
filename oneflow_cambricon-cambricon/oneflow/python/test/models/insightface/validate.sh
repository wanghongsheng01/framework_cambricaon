# !/bin/bash

MODEL_LOAD_DIR="./insightface_nhwc/"

INPUT_IMAGE="./images/dog.jpeg"

python3 insightface_val.py \
    --val_img_dir $INPUT_IMAGE \
    --model_load_dir $MODEL_LOAD_DIR


