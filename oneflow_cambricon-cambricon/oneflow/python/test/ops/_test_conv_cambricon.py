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
import test_global_storage

# flow.config.enable_debug_mode(True)


def flow_conv2d(x, filters, kernel_size, strides, padding, data_format):
    def make_job(
        input_shape, filters, ksize, strides, padding, data_format, dtype=flow.float32
    ):
        flow.config.enable_legacy_model_io(True)
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def conv2d_job(
            x: tp.Numpy.Placeholder(input_shape, dtype=flow.float32)
        ) -> tp.Numpy:
            weight_shape = (filters, kernel_size, kernel_size, x.shape[3])
            weight = flow.get_variable(
                "conv-weight",
                shape=weight_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=0, maxval=1),
            )
            flow.watch(weight, test_global_storage.Setter("weight"))
            ans = flow.nn.conv2d(
                x,
                weight,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=[1, 1],
                name="conv",
            )
            return ans

        check_point = flow.train.CheckPoint()
        check_point.init()
        return conv2d_job

    conv2d_job = make_job(
        x.shape, filters, kernel_size, strides, padding, data_format, dtype=flow.float32
    )
    y = conv2d_job(x)
    return y


def flow_conv2d_cpu(x, filters, kernel_size, strides, padding, data_format):
    def make_job(
        input_shape,
        weight_shape,
        filters,
        ksize,
        strides,
        padding,
        data_format,
        dtype=flow.float32,
    ):
        flow.config.enable_legacy_model_io(True)
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cpu", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def conv2d_job(
            x: tp.Numpy.Placeholder(input_shape, dtype=flow.float32),
            weight: tp.Numpy.Placeholder(weight_shape, dtype=flow.float32),
        ) -> tp.Numpy:
            # weight_shape = (filters, kernel_size, kernel_size, x.shape[3])
            # weight = flow.get_variable(
            #     "conv-weight",
            #     shape=weight_shape,
            #     dtype=flow.float,
            #     initializer=flow.random_uniform_initializer(minval=0, maxval=1),
            # )
            # flow.watch(weight, test_global_storage.Setter("weight"))
            ans = flow.nn.conv2d(
                x,
                weight,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=[1, 1],
                name="conv",
            )
            return ans

        check_point = flow.train.CheckPoint()
        check_point.init()
        return conv2d_job

    weight = test_global_storage.Get("weight")
    weight_shape = (filters, kernel_size, kernel_size, x.shape[3])
    conv2d_job = make_job(
        x.shape,
        weight_shape,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        dtype=flow.float32,
    )
    y = conv2d_job(x, weight)
    return y


def conv2d_compare_with_cpu(
    test_case, input_shape, filters, kernel_size, strides, padding, data_format
):
    x = np.random.random(input_shape).astype(np.float32)
    flow_res = flow_conv2d(x, filters, kernel_size, strides, padding, data_format)
    print(flow_res)
    print("===========")
    flow.clear_default_session()
    flow_cpu_res = flow_conv2d_cpu(
        x, filters, kernel_size, strides, padding, data_format
    )
    print(flow_cpu_res)
    test_case.assertTrue(np.allclose(flow_cpu_res, flow_res, rtol=1e-1, atol=1e-5))


@flow.unittest.skip_unless_1n1d()
class TestConv(flow.unittest.TestCase):
    def test_1(test_case):
        conv2d_compare_with_cpu(
            test_case,
            input_shape=(1, 224, 224, 3),
            filters=1,
            kernel_size=3,
            strides=2,
            padding="SAME",
            data_format="NHWC",
        )

    def test_2(test_case):
        conv2d_compare_with_cpu(
            test_case,
            input_shape=(3, 10, 10, 3),
            filters=1,
            kernel_size=3,
            strides=1,
            padding="SAME",
            data_format="NHWC",
        )

    def test_3(test_case):
        conv2d_compare_with_cpu(
            test_case,
            input_shape=(3, 11, 11, 3),
            filters=1,
            kernel_size=3,
            strides=1,
            padding="SAME",
            data_format="NHWC",
        )

    def test_4(test_case):
        conv2d_compare_with_cpu(
            test_case,
            input_shape=(1, 224, 224, 3),
            filters=1,
            kernel_size=3,
            strides=2,
            padding="VALID",
            data_format="NHWC",
        )

    def test_5(test_case):
        conv2d_compare_with_cpu(
            test_case,
            input_shape=(3, 10, 10, 3),
            filters=1,
            kernel_size=3,
            strides=1,
            padding="VALID",
            data_format="NHWC",
        )

    def test_6(test_case):
        conv2d_compare_with_cpu(
            test_case,
            input_shape=(3, 11, 11, 3),
            filters=1,
            kernel_size=3,
            strides=1,
            padding="VALID",
            data_format="NHWC",
        )


if __name__ == "__main__":
    unittest.main()
