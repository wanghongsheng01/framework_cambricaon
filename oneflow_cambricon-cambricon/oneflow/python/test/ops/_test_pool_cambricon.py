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
import tensorflow as tf

# flow.config.enable_debug_mode(True)


def flow_maxpool2d(x, ksize, strides, padding, data_format):
    if padding == "SAME":
        padding = "SAME_UPPER"

    def make_job(input_shape, ksize, strides, padding, data_format, dtype=flow.float32):
        flow.config.enable_legacy_model_io(True)
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def maxpool2d_job(
            x: tp.Numpy.Placeholder(input_shape, dtype=flow.float32)
        ) -> tp.Numpy:
            ans = flow.nn.max_pool2d(
                input=x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )
            return ans

        check_point = flow.train.CheckPoint()
        check_point.init()
        return maxpool2d_job

    maxpool2d_job = make_job(
        x.shape, ksize, strides, padding, data_format, dtype=flow.float32
    )
    y = maxpool2d_job(x)
    return y


def flow_avgpool2d(x, ksize, strides, padding, data_format):
    if padding == "SAME":
        padding = "SAME_UPPER"

    def make_job(input_shape, ksize, strides, padding, data_format, dtype=flow.float32):
        flow.config.enable_legacy_model_io(True)
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def avgpool2d_job(
            x: tp.Numpy.Placeholder(input_shape, dtype=flow.float32)
        ) -> tp.Numpy:
            ans = flow.nn.avg_pool2d(
                input=x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )
            return ans

        check_point = flow.train.CheckPoint()
        check_point.init()
        return avgpool2d_job

    avgpool2d_job = make_job(
        x.shape, ksize, strides, padding, data_format, dtype=flow.float32
    )
    y = avgpool2d_job(x)
    return y


def tf_maxpool2d(x, ksize, strides, padding, data_format):
    x_tf = tf.Variable(x)
    y_tf = tf.nn.max_pool2d(
        x_tf, ksize=ksize, strides=strides, padding=padding, data_format=data_format
    )
    return y_tf.numpy()


def tf_avgpool2d(x, ksize, strides, padding, data_format):
    x_tf = tf.Variable(x)
    y_tf = tf.nn.avg_pool2d(
        x_tf, ksize=ksize, strides=strides, padding=padding, data_format=data_format
    )
    return y_tf.numpy()


def maxpool2d_compare_with_torch(
    test_case, input_shape, ksize, strides, padding, data_format
):
    x = np.random.random(input_shape).astype(np.float32)
    x = (x - 0.5) * 10
    flow_res = flow_maxpool2d(x, ksize, strides, padding, data_format)
    tf_res = tf_maxpool2d(x, ksize, strides, padding, data_format)
    test_case.assertTrue(np.allclose(tf_res, flow_res))


def avgpool2d_compare_with_torch(
    test_case, input_shape, ksize, strides, padding, data_format
):
    x = np.random.random(input_shape).astype(np.float32)
    x = (x - 0.5) * 10
    flow_res = flow_avgpool2d(x, ksize, strides, padding, data_format)
    tf_res = tf_avgpool2d(x, ksize, strides, padding, data_format)
    test_case.assertTrue(np.allclose(tf_res, flow_res, rtol=1e-3, atol=1e-5))


@flow.unittest.skip_unless_1n1d()
class TestPool(flow.unittest.TestCase):
    def test_1(test_case):
        maxpool2d_compare_with_torch(
            test_case,
            input_shape=(1, 100, 100, 2),
            ksize=3,
            strides=2,
            padding="VALID",
            data_format="NHWC",
        )

    def test_2(test_case):
        avgpool2d_compare_with_torch(
            test_case,
            input_shape=(1, 100, 100, 2),
            ksize=3,
            strides=2,
            padding="SAME",
            data_format="NHWC",
        )

    def test_3(test_case):
        maxpool2d_compare_with_torch(
            test_case,
            input_shape=(1, 100, 100, 5),
            ksize=3,
            strides=2,
            padding="SAME",
            data_format="NHWC",
        )

    def test_4(test_case):
        avgpool2d_compare_with_torch(
            test_case,
            input_shape=(1, 100, 100, 2),
            ksize=3,
            strides=2,
            padding="VALID",
            data_format="NHWC",
        )


if __name__ == "__main__":
    unittest.main()
