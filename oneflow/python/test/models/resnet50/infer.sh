MODEL_LOAD_DIR="./resnet50_nhwc/"

INPUT_IMAGE="./images/fish.jpg"

python3 infer_resnet50.py \
    --input_image_path $INPUT_IMAGE \
    --model_load_dir $MODEL_LOAD_DIR