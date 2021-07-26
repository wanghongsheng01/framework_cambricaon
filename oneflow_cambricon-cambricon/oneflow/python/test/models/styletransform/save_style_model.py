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
import os
import shutil
import argparse
import oneflow as flow
import style_model


def _init_oneflow_env_and_config():
    flow.env.init()
    flow.enable_eager_execution(False)
    flow.config.enable_legacy_model_io(True)


def _make_style_transform_predict_func(width, height, backend="gpu"):
    batch_size = 1
    channels = 3

    func_cfg = flow.function_config()
    func_cfg.default_placement_scope(flow.scope.placement(backend, "0:0"))

    @flow.global_function("predict", function_config=func_cfg)
    def predict_fn(
        image: flow.typing.Numpy.Placeholder(
            shape=(batch_size, height, width, channels), dtype=flow.float32
        )
    ) -> flow.typing.Numpy:
        style_out = style_model.styleNet(image, backend=backend)
        return style_out

    return predict_fn


def main(args):
    _init_oneflow_env_and_config()

    predict_fn = _make_style_transform_predict_func(args.image_width, args.image_height, args.backend)
    flow.train.CheckPoint().load(args.model_dir)
    # flow.load_variables(flow.checkpoint.get(args.model_dir))
    print("predict_fn construct finished")

    saved_model_path = args.save_dir
    model_version = args.model_version

    model_version_path = os.path.join(saved_model_path, str(model_version))
    if os.path.exists(model_version_path) and os.path.isdir(model_version_path):
        if args.force_save:
            print(
                f"WARNING: The model version path '{model_version_path}' already exist"
                ", old version directory will be replaced"
            )
            shutil.rmtree(model_version_path)
        else:
            raise ValueError(
                f"The model version path '{model_version_path}' already exist"
            )

    saved_model_builder = (
        flow.saved_model.ModelBuilder(saved_model_path)
        .ModelName(args.model_name)
        .Version(model_version)
    )
    saved_model_builder.AddFunction(predict_fn).Finish()
    saved_model_builder.Save()


def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        help="gpu or cambricon"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="stylenet_nhwc",
        help="model parameters directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="style_transform_models",
        help="directory to save models",
    )
    parser.add_argument(
        "--model_name", type=str, default="style_transform", help="model name"
    )
    parser.add_argument("--model_version", type=int, default=1, help="model version")
    parser.add_argument(
        "--force_save",
        default=False,
        action="store_true",
        help="force save model whether already exists or not",
    )
    parser.add_argument(
        "--image_width", type=int, default=640, help="input image width"
    )
    parser.add_argument(
        "--image_height", type=int, default=640, help="input image height"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
