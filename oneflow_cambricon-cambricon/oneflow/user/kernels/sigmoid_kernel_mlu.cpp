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

typedef struct Sigmoid_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlActivationDescriptor_t activation_desc = nullptr;
} Sigmoid;

template<DeviceType device_type, CamDataType T>
class SigmoidKernelCambricon final : public user_op::OpKernel {
 public:
  SigmoidKernelCambricon() = default;
  ~SigmoidKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);

    cnnlActivationMode_t mode = CNNL_ACTIVATION_SIGMOID;
    cnnlNanPropagation_t sigmoid_nan_opt = CNNL_NOT_PROPAGATE_NAN;
    float coef = 1.0;
    Sigmoid sigmoid;
    sigmoidDataType datainfo;
    datainfo.input_dtype = convert(T);
    datainfo.output_dtype = convert(T);

    set_tensor_desc(sigmoid.input_desc, x->shape().At(0), x->shape().At(1), x->shape().At(2),
                    x->shape().At(3), datainfo.input_dtype, datainfo.layout);
    set_tensor_desc(sigmoid.output_desc, y->shape().At(0), y->shape().At(1), y->shape().At(2),
                    y->shape().At(3), datainfo.output_dtype, datainfo.layout);

    void* x_ptr = (void*)x->dptr();
    void* y_ptr = (void*)y->dptr();

    CNNL_CHECK(cnnlCreateActivationDescriptor(&sigmoid.activation_desc));
    CNNL_CHECK(cnnlSetActivationDescriptor(sigmoid.activation_desc, mode, sigmoid_nan_opt, coef));
    CNNL_CHECK(cnnlActivationForward(ctx->device_ctx()->cambricon_handle(), sigmoid.activation_desc,
                                     nullptr, sigmoid.input_desc, x_ptr, nullptr,
                                     sigmoid.output_desc, y_ptr));
    CNNL_CHECK(cnnlDestroyActivationDescriptor(sigmoid.activation_desc));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("sigmoid")                           \
      .SetCreateFn<SigmoidKernelCambricon<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));

#ifdef WITH_CAMBRICON
REGISTER_SIGMOID_KERNEL(DeviceType::kCambricon, CamDataType::kFLOAT32)
#endif

}  // namespace oneflow

#endif
