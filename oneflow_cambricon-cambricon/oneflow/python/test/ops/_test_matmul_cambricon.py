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


def flow_matmul(a, b):
    def make_job(a_shape, b_shape, dtype=flow.float32):
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))
        flow.config.enable_legacy_model_io(True)

        @flow.global_function(type="predict", function_config=config)
        def matmul_job(
            a: tp.Numpy.Placeholder(a_shape), b: tp.Numpy.Placeholder(b_shape)
        ) -> tp.Numpy:
            return flow.matmul(a, b, transpose_b=True)

        return matmul_job

    matmul_fakedev_job = make_job(a.shape, b.shape, dtype=flow.float32)
    c = matmul_fakedev_job(a, b)
    return c


def np_matmul(a, b):
    b = np.transpose(b, (1, 0))
    return np.matmul(a, b)


def _compare_with_np(test_case, a_shape, b_shape):
    a = np.random.random(a_shape).astype(np.float32)
    b = np.random.random(b_shape).astype(np.float32)
    np_res = np_matmul(a, b)
    print(np_res)
    flow_res = flow_matmul(a, b)
    print(flow_res)
    test_case.assertTrue(np.allclose(np_res, flow_res, rtol=1e-1, atol=1e-5))


@flow.unittest.skip_unless_1n1d()
class TestMatmul(flow.unittest.TestCase):
    def test_random_value(test_case):
        _compare_with_np(test_case, (1, 2048), (1000, 2048))


if __name__ == "__main__":
    unittest.main()
