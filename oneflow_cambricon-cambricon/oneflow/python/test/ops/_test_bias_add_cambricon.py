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

# flow.config.enable_debug_mode(True)
flow.config.enable_legacy_model_io(True)


def flow_bias_add(x, y):
    def make_job(input_shape, bias_shape, dtype=flow.float32):
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def bias_add_job(
            x: tp.Numpy.Placeholder(input_shape), y: tp.Numpy.Placeholder(bias_shape)
        ) -> tp.Numpy:
            return flow.nn.bias_add(x, y, data_format="NHWC")

        return bias_add_job

    bias_add_job = make_job(x.shape, y.shape, dtype=flow.float32)
    y = bias_add_job(x, y)
    return y


def np_bias_add(x, y):
    y = np.reshape(y, newshape=(1, y.shape[0]))
    return x + y


def _compare_with_np(test_case, input_shape, bias_shape):
    x = np.random.random(input_shape).astype(np.float32)
    x = (x - 0.5) * 10
    y = np.random.random(bias_shape).astype(np.float32)
    np_res = np_bias_add(x, y)
    flow_res = flow_bias_add(x, y)
    test_case.assertTrue(np.allclose(np_res, flow_res))


@flow.unittest.skip_unless_1n1d()
class TestBiasAdd(flow.unittest.TestCase):
    def test_random_value(test_case):
        _compare_with_np(test_case, (1, 1000), (1000,))


if __name__ == "__main__":
    unittest.main()
