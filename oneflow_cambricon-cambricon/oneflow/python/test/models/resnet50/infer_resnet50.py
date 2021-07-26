"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
import oneflow.typing as tp

import argparse
import cv2
import numpy as np

from imagenet1000_clsidx_to_labels import clsidx_2_labels
from resnet50_model import resnet50

def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
    im = (im - [123.68, 116.779, 103.939]) / [58.393, 57.12, 57.375]
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


flow.config.enable_legacy_model_io(True)


def main(args):
    input_image = load_image(args.input_image_path)
    height = 224
    width = 224
    flow.env.init()
    config = flow.function_config()
    config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

    @flow.global_function("predict", function_config=config)
    def InferenceNet(
        images: tp.Numpy.Placeholder((1, height, width, 3), dtype=flow.float)
    ) -> tp.Numpy:
        logits = resnet50(images, args, training=False)
        predictions = flow.nn.softmax(logits)
        return predictions

    print("===============================>load begin")
    flow.train.CheckPoint().load(args.model_load_dir)
    print("===============================>load end")

    import datetime

    a = datetime.datetime.now()

    print("predict begin")
    reset_out = InferenceNet(input_image)
    print("predict end")
    clsidx = reset_out.argmax()

    b = datetime.datetime.now()
    c = b - a

    print("time: %s ms, height: %d, width: %d" % (c.microseconds / 1000, height, width))
    print(
        "resnet50 predict prob %f, class %s"
        % (reset_out.max(), clsidx_2_labels[clsidx])
    )


def get_parser(parser=None):
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser("flags for neural style")
    parser.add_argument(
        "--input_image_path", type=str, default="images/tiger.jpg", help="image path"
    )
    parser.add_argument(
        "--model_load_dir", type=str, default="", help="model save directory"
    )
    parser.add_argument(
        "--channel_last",
        type=str2bool,
        default=True,
        help="Whether to use use channel last mode(nhwc)",
    )
    # fuse bn relu or bn add relu
    parser.add_argument(
        "--fuse_bn_relu",
        type=str2bool,
        default=False,
        help="Whether to use use fuse batch normalization relu. Currently supported in origin/master of OneFlow only.",
    )
    parser.add_argument(
        "--fuse_bn_add_relu",
        type=str2bool,
        default=False,
        help="Whether to use use fuse batch normalization add relu. Currently supported in origin/master of OneFlow only.",
    )
    parser.add_argument(
        "--pad_output",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to pad the output to number of image channels to 4.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
