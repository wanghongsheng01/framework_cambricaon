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
typedef struct Softmax_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlSoftmaxAlgorithm_t algorithm = CNNL_SOFTMAX_ACCURATE;
  cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  float hw_time = 0;
} Softmax;

template<DeviceType device_type>
class SoftmaxKernelCambricon final : public user_op::OpKernel {
 public:
  SoftmaxKernelCambricon() = default;
  ~SoftmaxKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);

    Softmax softmax;
    SoftmaxType datainfo;
    datainfo.input_dtype = convert(CamDataType::kFLOAT32);
    datainfo.output_dtype = convert(CamDataType::kFLOAT32);
    CHECK_EQ(x->shape().NumAxes(), 2)
        << "The number of axes of softmax op input shape should equal to 2!";
    set_tensor_desc_softmax(softmax.input_desc, x->shape().At(0), x->shape().At(1),
                            datainfo.input_dtype, datainfo.layout);
    set_tensor_desc_softmax(softmax.output_desc, y->shape().At(0), y->shape().At(1),
                            datainfo.output_dtype, datainfo.layout);

    const void* x_ptr = (const void*)x->dptr();
    void* y_ptr = (void*)y->dptr();

    CNNL_CHECK(cnnlSoftmaxForward(ctx->device_ctx()->cambricon_handle(), softmax.algorithm,
                                  softmax.mode, nullptr, softmax.input_desc, x_ptr, nullptr,
                                  softmax.output_desc, y_ptr));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_KERNEL(device, dtype)                                                  \
  REGISTER_USER_KERNEL("softmax")                                                               \
      .SetCreateFn<SoftmaxKernelCambricon<device>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("in", 0) == dtype))                              \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      })

REGISTER_SOFTMAX_KERNEL(DeviceType::kCambricon, DataType::kFloat);

}  // namespace

}  // namespace oneflow

#endif
