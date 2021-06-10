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
#include "oneflow/core/framework/device.h"

namespace oneflow {

Maybe<const Device> DeviceInferFn(user_op::DeviceInferContext* ctx) {
  *ctx->OutputTensorDevice4ArgNameAndIndex("out", 0) =
      ctx->InputTensorDevice4ArgNameAndIndex("in", 0);
  return Device::New("cuda_d2d");
}

REGISTER_USER_OP("eager_nccl_all_reduce")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .Broadcast(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
