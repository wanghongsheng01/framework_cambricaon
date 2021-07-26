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
from oneflow.python.test.ops.test_util import GenArgList


def compare_with_np(device_type, x_shape, y_shape, axis):
    def _np_concat(x, y):
        return np.concatenate((x, y), axis)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_placement_scope(flow.scope.placement(device_type, "0:0"))
    flow.config.enable_legacy_model_io(True)

    @flow.global_function(type="predict", function_config=func_config)
    def ConcatJob(
        x: tp.Numpy.Placeholder(shape=x_shape, dtype=flow.float32),
        y: tp.Numpy.Placeholder(shape=y_shape, dtype=flow.float32),
    ) -> tp.Numpy:
        x_var = flow.get_variable(
            "x",
            shape=x_shape,
            dtype=flow.float32,
            initializer=flow.constant_initializer(0),
            trainable=False,
        )
        x_var = x_var + x

        y_var = flow.get_variable(
            "y",
            shape=y_shape,
            dtype=flow.float32,
            initializer=flow.constant_initializer(0),
            trainable=False,
        )
        y_var = y_var + y

        out = flow.concat([x_var, y_var], axis)
        return out

    check_point = flow.train.CheckPoint()
    check_point.init()
    data_x = np.random.random(size=x_shape)
    data_y = np.random.random(size=y_shape)
    of_out = ConcatJob(data_x, data_y)
    np_out = _np_concat(data_x, data_y)
    assert np.allclose(of_out, np_out, rtol=1e-4, atol=1e-4)


@flow.unittest.skip_unless_1n1d()
class TestTranspose(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cambricon"]
        arg_dict["x_shape"] = [(10, 20, 30, 40)]
        arg_dict["y_shape"] = [(10, 20, 30, 40)]
        arg_dict["axis"] = [0, 1, 2, 3]
        for arg in GenArgList(arg_dict):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
