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
#ifdef WITH_CAMBRICON

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/mlu_tools.h"
#include "cnrt.h"
#include "cnnl.h"
#include <stdio.h>

namespace oneflow {

namespace {

typedef struct Batchnorm_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlTensorDescriptor_t weight_bias_mean_var_desc = nullptr;
} Batchnorm;

void batchnorm2d(const void* input_, const void* weight_, const void* bias_, const void* mean_,
                 const void* var_, void* output_, CamDataType input_type,
                 CamDataType weight_bias_mean_var_type, CamDataType output_type, int input_dim_n,
                 int input_dim_h, int input_dim_w, int input_dim_c, float eps, cnnlHandle_t handle,
                 user_op::KernelComputeContext* ctx) {
  Batchnorm bn;

  BatchNormType datainfo;
  datainfo.input_dtype = convert(input_type);
  datainfo.output_dtype = convert(output_type);
  datainfo.weight_bias_mean_var_desc_dtype = convert(weight_bias_mean_var_type);

  set_tensor_desc(bn.input_desc, input_dim_n, input_dim_h, input_dim_w, input_dim_c,
                  datainfo.input_dtype, datainfo.layout);
  set_tensor_desc(bn.output_desc, input_dim_n, input_dim_h, input_dim_w, input_dim_c,
                  datainfo.output_dtype, datainfo.layout);
  set_tensor_desc_batchnorm(bn.weight_bias_mean_var_desc, input_dim_c,
                            datainfo.weight_bias_mean_var_desc_dtype, datainfo.layout);
  CNNL_CHECK(cnnlBatchNormForwardInference(handle, nullptr, nullptr, bn.input_desc, input_,
                                           bn.weight_bias_mean_var_desc, weight_, bias_, mean_,
                                           var_, eps, bn.output_desc, output_));
  ctx->device_ctx()->SyncDevice();
}

class BnFp32CambriconKernel final : public user_op::OpKernel {
 public:
  BnFp32CambriconKernel() = default;
  ~BnFp32CambriconKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(!training);
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    CHECK_EQ(axis, x->shape().NumAxes() - 1);
    const auto epsilon = ctx->Attr<float>("epsilon");
    int n = 0, c = 0, h = 0, w = 0;
    if (x->shape().NumAxes() == 2) {
      n = x->shape().At(0);
      h = 1;
      w = 1;
      c = x->shape().At(1);
    } else {
      n = x->shape().At(0);
      h = x->shape().At(1);
      w = x->shape().At(2);
      c = x->shape().At(3);
    }

    batchnorm2d(x->raw_dptr(), gamma->raw_dptr(), beta->raw_dptr(), moving_mean->raw_dptr(),
                moving_variance->raw_dptr(), y->mut_raw_dptr(), CamDataType::kFLOAT32,
                CamDataType::kFLOAT32, CamDataType::kFLOAT32, n, h, w, c, epsilon,
                ctx->device_ctx()->cambricon_handle(), ctx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("normalization")
    .SetCreateFn<BnFp32CambriconKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cambricon")
                     & (user_op::HobDataType("y", 0) == DataType::kFloat)
                     & (user_op::HobAttr<bool>("training") == false));

}  // namespace

}  // namespace oneflow

#endif
