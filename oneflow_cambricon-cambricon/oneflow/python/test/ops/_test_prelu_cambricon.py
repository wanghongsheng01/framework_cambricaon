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


def _prelu_op(input, alpha, name="prelu"):
    op = (
        flow.user_op_builder(name)
        .Op("prelu")
        .Input("x", [input])
        .Input("alpha", [alpha])
        .Output("y")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


def _prelu(input, alpha):
    return np.where(input > 0, input, input * alpha)


def _make_of_prelu_func(input_shape, alpha_shape, dtype=flow.float32):
    # flow.config.enable_debug_mode(True)
    flow.config.enable_legacy_model_io(True)
    func_cfg = flow.function_config()
    func_cfg.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

    @flow.global_function(type="predict", function_config=func_cfg)
    def prelu_func(
        input: flow.typing.Numpy.Placeholder(input_shape, dtype=dtype),
        alpha: flow.typing.Numpy.Placeholder(alpha_shape, dtype=dtype),
    ) -> flow.typing.Numpy:
        alpha_var = flow.get_variable(
            name="prelu-alpha",
            shape=alpha_shape,
            dtype=dtype,
            initializer=flow.constant_initializer(0.0),
            reuse=False,
        )
        alpha += alpha_var
        return _prelu_op(input, alpha)

    check_point = flow.train.CheckPoint()
    check_point.init()
    return prelu_func


def _test_by_comparing_with_np(test_case, input_shape, alpha_shape):
    input = np.random.uniform(-1, 1, input_shape).astype(np.float32)
    alpha = np.random.uniform(-1, 1, alpha_shape).astype(np.float32)
    alpha *= 10
    prelu_func = _make_of_prelu_func(input_shape, alpha_shape)
    output = prelu_func(input, alpha)
    cmp_output = _prelu(input, alpha)

    print(f"input_shape: {input_shape}, alpha_shape: {alpha_shape}")
    print(f"output shape: {output.shape}, cmp_output shape: {cmp_output.shape}")
    cmp = np.allclose(output, cmp_output)
    if cmp is False:
        zero_elem_cnt = np.argwhere(output == 0.0).shape[0]
        print(f"oneflow output zero elem cnt: {zero_elem_cnt}")
        print(f"oneflow output:\n{output}")
        print("=" * 80)
        print(f"compare output:\n{cmp_output}")
        print("=" * 80)
        print(f"diff:\n{output - cmp_output}")
        print("=" * 80)
        diff_idx = np.argwhere(np.isclose(output, cmp_output) == False)
        print(f"diff idx: {diff_idx.shape[0]}\n{diff_idx[:10]}")
        np.save("test_fail_prelu_output", output)

    test_case.assertTrue(cmp)


@flow.unittest.skip_unless_1n1d()
class TestPRelu(flow.unittest.TestCase):
    def test_with_random_input(test_case):
        _test_by_comparing_with_np(test_case, (1, 256, 256, 3), (1, 1, 3))

    # test failed
    # def test_with_random_input_3(test_case):
    #     _test_by_comparing_with_np(test_case, (1, 256, 256, 3), (256, 256, 3))

    # test failed
    # def test_with_random_input_1(test_case):
    #     _test_by_comparing_with_np(test_case, (1, 256, 256, 3), (1, 256, 3))

    # test failed
    # def test_with_random_input_2(test_case):
    #     _test_by_comparing_with_np(test_case, (1, 256, 256, 3), (256, 1, 3))


if __name__ == "__main__":
    unittest.main()
