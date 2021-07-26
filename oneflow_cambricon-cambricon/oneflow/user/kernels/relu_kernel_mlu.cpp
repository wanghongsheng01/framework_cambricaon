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

typedef struct Activation_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlActivationDescriptor_t activation_desc = nullptr;
  float hw_time = 0;
} Activation;

template<DeviceType device_type>
class ReluKernelCambricon final : public user_op::OpKernel {
 public:
  ReluKernelCambricon() = default;
  ~ReluKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);

    cnnlActivationMode_t mode = CNNL_ACTIVATION_RELU;
    cnnlNanPropagation_t relu_nan_opt = CNNL_NOT_PROPAGATE_NAN;
    float coef = 1.0;

    Activation relu;
    reluDataType datainfo;
    datainfo.input_dtype = convert(CamDataType::kFLOAT32);
    datainfo.output_dtype = convert(CamDataType::kFLOAT32);

    set_tensor_desc(relu.input_desc, x->shape().At(0), x->shape().At(1), x->shape().At(2),
                    x->shape().At(3), datainfo.input_dtype, datainfo.layout);
    set_tensor_desc(relu.output_desc, y->shape().At(0), y->shape().At(1), y->shape().At(2),
                    y->shape().At(3), datainfo.output_dtype, datainfo.layout);

    CNNL_CHECK(cnnlCreateActivationDescriptor(&relu.activation_desc));
    CNNL_CHECK(cnnlSetActivationDescriptor(relu.activation_desc, mode, relu_nan_opt, coef));

    void* x_ptr = (void*)x->dptr();
    void* y_ptr = (void*)y->dptr();

    CNNL_CHECK(cnnlActivationForward(ctx->device_ctx()->cambricon_handle(), relu.activation_desc,
                                     nullptr, relu.input_desc, x_ptr, nullptr, relu.output_desc,
                                     y_ptr));
    CNNL_CHECK(cnnlDestroyActivationDescriptor(relu.activation_desc));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_KERNEL(device, dtype)                                                     \
  REGISTER_USER_KERNEL("relu")                                                                  \
      .SetCreateFn<ReluKernelCambricon<device>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("in", 0) == dtype))                              \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      })

#ifdef WITH_CAMBRICON
REGISTER_RELU_KERNEL(DeviceType::kCambricon, DataType::kFloat);
#endif

}  // namespace

}  // namespace oneflow

#endif
