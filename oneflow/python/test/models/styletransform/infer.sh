MODEL_LOAD_DIR="./stylenet_nhwc/"

INPUT_IMAGE="./images/content-images/amber.jpg"
OUTPUT_IMAGE="./images/style_out_amber_nhwc.jpg"

BACKEND=${1:-gpu}

python3 infer_of_neural_style.py \
    --backend $BACKEND \
    --input_image_path $INPUT_IMAGE \
    --output_image_path $OUTPUT_IMAGE \
    --model_load_dir $MODEL_LOAD_DIR