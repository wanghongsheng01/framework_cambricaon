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
from oneflow.python.test.ops.test_util import GenArgList


def compare_with_np(device_type, input_shape, perm):
    def _np_transpose(x):
        return np.transpose(x, perm)

    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_placement_scope(flow.scope.placement(device_type, "0:0"))

    flow.config.enable_legacy_model_io(True)

    @flow.global_function(type="predict", function_config=func_config)
    def TransposeJob(
        x: tp.Numpy.Placeholder(shape=input_shape, dtype=flow.float32)
    ) -> tp.Numpy:
        x_var = flow.get_variable(
            "input",
            shape=input_shape,
            dtype=flow.float,
            initializer=flow.constant_initializer(0),
            trainable=False,
        )
        x_var = x_var + x
        out = flow.transpose(x, perm)
        return out

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    data_x = np.random.random(size=input_shape)
    np_y = _np_transpose(data_x)
    of_y = TransposeJob(data_x)
    assert np.allclose(np_y, of_y, rtol=1e-5, atol=1e-5)


@flow.unittest.skip_unless_1n1d()
class TestTranspose(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cambricon"]
        arg_dict["input_shape"] = [(10, 224, 224, 13)]
        arg_dict["perm"] = [(2, 0, 1, 3), (1, 0, 2, 3), (3, 2, 1, 0), (3, 1, 2, 0)]
        for arg in GenArgList(arg_dict):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
