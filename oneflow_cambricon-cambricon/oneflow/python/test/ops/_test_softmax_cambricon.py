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
import test_global_storage

# flow.config.enable_debug_mode(True)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def flow_softmax(x):
    def make_job(input_shape, dtype=flow.float32):
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

        flow.config.enable_legacy_model_io(True)

        @flow.global_function(type="predict", function_config=config)
        def softmax_job(x: tp.Numpy.Placeholder(input_shape)) -> tp.Numpy:
            return flow.nn.softmax(x)

        return softmax_job

    softmax_fakedev_job = make_job(x.shape, dtype=flow.float32)
    y = softmax_fakedev_job(x)
    return y


def tf_softmax(x):
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(tf.constant(x), name="x")
        tf_out = tf.nn.softmax(x, axis=1)
    return tf_out.numpy()


def _compare_with_np(test_case, input_shape):
    x = np.random.random(input_shape).astype(np.float32)
    tf_res = tf_softmax(x)
    print("tf res: ", tf_res)
    flow_res = flow_softmax(x)
    print("flow_res: ", flow_res)
    test_case.assertTrue(np.allclose(tf_res, flow_res, rtol=1e-03, atol=1e-04))


@flow.unittest.skip_unless_1n1d()
class TestSoftmax(flow.unittest.TestCase):
    def test_random_value(test_case):
        _compare_with_np(test_case, (5, 3))


if __name__ == "__main__":
    unittest.main()
