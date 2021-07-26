import math
import os
import argparse
import numpy as np
import cv2
import oneflow as flow

import fresnet100
import oneflow.typing as tp
from typing import Tuple
from scipy.spatial import distance

def get_val_args():
    val_parser = argparse.ArgumentParser(description="flags for validation")
    val_parser.add_argument(
            "--val_img_dir",
            type=str,
            default="./woman.jpeg",
            help="validation dataset dir",
        )

    # distribution config
    val_parser.add_argument(
        "--device_num_per_node",
        type=int,
        default=1,
        required=False,
    )
    val_parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="node/machine number for training",
    )

    val_parser.add_argument(
        "--val_batch_size",
        default=1,
        type=int,
        help="validation batch size totally",
    )
    # model and log
    val_parser.add_argument(
        "--log_dir", type=str, default="./log", help="log info save"
    )
    val_parser.add_argument(
        "--model_load_dir", default="/insightface_nhwc", help="path to load model."
    )
    return val_parser.parse_args()


def load_image(image_path):
    im = cv2.imread(image_path)
    dsize = (112, 112)
    rgb_mean = [127.5, 127.5, 127.5]
    std_values = [128.0, 128.0, 128.0]

    im = cv2.resize(im, dsize, interpolation = cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = (im - rgb_mean) / std_values
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 2, 3, 1))
    print("image size: ", im.shape)
    return np.ascontiguousarray(im, 'float32')

def get_cambricon_config():
    val_config = flow.function_config()
    val_config.default_logical_view(flow.scope.consistent_view())
    val_config.default_data_type(flow.float)
    val_config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))
    return val_config

def validation_job(images, config):
    @flow.global_function(type="predict", function_config=config)
    def get_symbol_val_job(
            images: flow.typing.Numpy.Placeholder(
                (1, 112, 112, 3)
            )
        ):
        print("val batch data: ", images.shape)
        embedding = fresnet100.get_symbol(images)
        return embedding

    return get_symbol_val_job

def do_validation(images, val_job, name_suffix):
    print("Validation starts...")
    batch_size = 1
    total_images_num = 1

    _em = val_job(images).get()
    return _em


def load_checkpoint(model_load_dir):
    print("=" * 20 + " model load begin " + "=" * 20)
    flow.train.CheckPoint().load(model_load_dir)
    print("=" * 20 + " model load end " + "=" * 20)


def main():
    args = get_val_args()
    flow.env.init()
    flow.env.log_dir(args.log_dir)
    # validation
    print("args: ", args)
    output_list = [] 
    if os.path.exists(args.val_img_dir):
        print("=" * 20 + " image load begin " + "=" * 20)
        images = load_image(args.val_img_dir)
        print("=" * 20 + " image load end " + "=" * 20)
    else: 
        raise ValueError ("Image path for validation does NOT exist!")
    flow.config.enable_legacy_model_io(True)
    val_job = validation_job(images, get_cambricon_config())
    load_checkpoint(args.model_load_dir)
    print("=" * 20 + " Prediction begins " + "=" * 20)   
    mlu_res = do_validation(images, val_job, "mlu")
    print("=" * 20 + " Prediction ends " + "=" * 20)
    flow.clear_default_session()

if __name__ == "__main__":
    main()