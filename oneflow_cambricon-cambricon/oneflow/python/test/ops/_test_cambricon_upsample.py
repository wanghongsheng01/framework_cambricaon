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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.typing as tp

flow.config.enable_legacy_model_io(True)


def flow_upsample(x, input_shape, size, data_format, interpolation):
    def make_job(input_shape, size, data_format, interpolation, dtype=flow.float32):
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def upsample_job(x: tp.Numpy.Placeholder(input_shape)) -> tp.Numpy:
            return flow.layers.upsample_2d(
                x, size=size, data_format=data_format, interpolation=interpolation
            )

        return upsample_job

    upsample_fakedev_job = make_job(
        x.shape, size, data_format, interpolation, dtype=flow.float32
    )
    y = upsample_fakedev_job(x)
    return y


def _compare_with_np(test_case, input_shape, size, data_format, interpolation):
    x = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]])
    flow_res = flow_upsample(x, input_shape, size, data_format, interpolation)
    print(flow_res)


@flow.unittest.skip_unless_1n1d()
class TestUpsample(flow.unittest.TestCase):
    def test_upsample(test_case):
        _compare_with_np(test_case, (1, 3, 3, 1), (2, 2), "NHWC", "bilinear")


if __name__ == "__main__":
    unittest.main()
