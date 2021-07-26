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
import unittest

import numpy as np
import oneflow as flow
import oneflow.typing as tp


def np_var(input_nhwc, eps=1e-05):
    assert len(input_nhwc.shape) == 4
    input_nhwc_reshape_to_1d = np.reshape(
        input_nhwc, (input_nhwc.shape[0], -1, input_nhwc.shape[3])
    )

    # compute instance normalization in numpy
    mean_np = np.mean(input_nhwc_reshape_to_1d, axis=(1), keepdims=True)
    in_sub_mean = input_nhwc_reshape_to_1d - mean_np
    var_np = np.var(input_nhwc_reshape_to_1d, axis=(1), keepdims=True)

    gamma = np.ones((1, 1, input_nhwc_reshape_to_1d.shape[2]), dtype=np.float32)
    beta = np.zeros((1, 1, input_nhwc_reshape_to_1d.shape[2]), dtype=np.float32)

    invar_np = 1.0 / np.sqrt(var_np + eps)
    out_np = in_sub_mean * invar_np * gamma + beta

    return out_np, mean_np, var_np


def _compare_with_np(test_case, input_shape, eps=1e-5):
    flow.config.enable_legacy_model_io(True)

    np_input = np.random.random(input_shape).astype(np.float32)

    config = flow.function_config()
    config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

    @flow.global_function(type="predict", function_config=config)
    def instance_norm_2d_job(
        x: tp.Numpy.Placeholder(np_input.shape, dtype=flow.float32)
    ):
        out, mean, var = flow.nn.InstanceNorm2d(x, eps=eps)
        return out, mean, var

    check_point = flow.train.CheckPoint()
    check_point.init()

    out_of, mean, var = instance_norm_2d_job(np_input).get()
    out_of = out_of.numpy()
    print(mean.numpy())
    print(var.numpy())
    out_np, mean_np, var_np = np_var(np_input, eps=eps)
    print("numpy: ")
    print(mean_np)
    print(var_np)

    test_case.assertTrue(np.allclose(out_np.flatten(), out_of.flatten(), atol=1e-04))


@flow.unittest.skip_unless_1n1d()
class TestInstanceNorm2D(flow.unittest.TestCase):
    def test_random_value(test_case):
        # _compare_with_np(test_case, (2, 2, 2, 3))
        _compare_with_np(test_case, (2, 512, 512, 3))


if __name__ == "__main__":
    unittest.main()
