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
import numpy as np
import argparse
import cv2

import oneflow as flow
import oneflow.typing as tp
import style_model


def float_list(x):
    return list(map(float, x.split(",")))


def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 2, 3, 1))
    return np.ascontiguousarray(im, "float32")


def recover_image(im):
    im = np.squeeze(im)
    print(im.shape)
    # im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)


flow.config.enable_legacy_model_io(True)


def main(args):
    input_image = load_image(args.input_image_path)
    height = input_image.shape[1]
    width = input_image.shape[2]
    flow.env.init()
    config = flow.function_config()
    config.default_placement_scope(flow.scope.placement(args.backend, "0:0"))

    @flow.global_function("predict", function_config=config)
    def PredictNet(
        image: tp.Numpy.Placeholder((1, height, width, 3), dtype=flow.float32)
    ) -> tp.Numpy:
        style_out = style_model.styleNet(image, trainable=True, backend=args.backend)
        return style_out

    print("===============================>load begin")
    # flow.load_variables(flow.checkpoint.get(args.model_load_dir))
    flow.train.CheckPoint().load(args.model_load_dir)
    print("===============================>load end")

    import datetime

    a = datetime.datetime.now()

    print("predict begin")
    style_out = PredictNet(input_image)
    style_out = np.clip(style_out, 0, 255)
    print("predict end")

    b = datetime.datetime.now()
    c = b - a

    print("time: %s ms, height: %d, width: %d" % (c.microseconds / 1000, height, width))

    cv2.imwrite(args.output_image_path, recover_image(style_out))
    # flow.checkpoint.save("./stylenet")


def get_parser(parser=None):
    parser = argparse.ArgumentParser("flags for neural style")
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        help="gpu or cambricon"
    )
    parser.add_argument(
        "--input_image_path", type=str, default="test_img/tiger.jpg", help="image path"
    )
    parser.add_argument(
        "--output_image_path", type=str, default="test_img/tiger.jpg", help="image path"
    )
    parser.add_argument(
        "--model_load_dir", type=str, default="", help="model save directory"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
