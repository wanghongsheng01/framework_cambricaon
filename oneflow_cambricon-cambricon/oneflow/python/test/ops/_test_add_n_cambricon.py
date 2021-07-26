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
from typing import Tuple

# flow.config.enable_debug_mode(True)
flow.config.enable_legacy_model_io(True)


def flow_add(x, input_shape, num_inputs):
    def make_job(input_shape, num_inputs, dtype=flow.float32):
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def add_job(xs: Tuple[(tp.Numpy.Placeholder(input_shape),) * num_inputs]):
            return flow.math.add_n(xs)

        return add_job

    add_fakedev_job = make_job(input_shape, num_inputs, dtype=flow.float32)
    y = add_fakedev_job(x).get().numpy()
    return y


def np_add(x, input_shape, inputs_num):
    ans = np.zeros(input_shape)
    for i in range(inputs_num):
        ans = ans + x[i]
    return ans


def _compare_with_np(test_case, input_shape, inputs_num):
    x = tuple(
        np.random.random(input_shape).astype(np.float32) for i in range(inputs_num)
    )
    np_res = np_add(x, input_shape, inputs_num)
    print(np_res)
    flow_res = flow_add(x, input_shape, inputs_num)
    print(flow_res)
    test_case.assertTrue(np.allclose(np_res, flow_res))


@flow.unittest.skip_unless_1n1d()
class TestRelu(flow.unittest.TestCase):
    def test_random_value(test_case):
        _compare_with_np(test_case, (1, 100, 100, 32), 10)


if __name__ == "__main__":
    unittest.main()
