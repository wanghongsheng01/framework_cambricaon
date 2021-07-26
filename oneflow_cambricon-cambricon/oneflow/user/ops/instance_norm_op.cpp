/*
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
*/
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

REGISTER_USER_OP("instance_norm_2d")
    .Input("in")
    .Input("gamma")
    .Input("beta")
    .Output("out")
    .Output("mean")
    .Output("var")
    .Attr<float>("eps", 1e-05)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      CHECK_EQ(in_shape->NumAxes(), 4);

      *ctx->Shape4ArgNameAndIndex("out", 0) = *in_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);

      *ctx->Shape4ArgNameAndIndex("mean", 0) = Shape({in_shape->At(0), 1, in_shape->At(3)});
      *ctx->Dtype4ArgNameAndIndex("mean", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);

      *ctx->Shape4ArgNameAndIndex("var", 0) = Shape({in_shape->At(0), 1, in_shape->At(3)});
      *ctx->Dtype4ArgNameAndIndex("var", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow