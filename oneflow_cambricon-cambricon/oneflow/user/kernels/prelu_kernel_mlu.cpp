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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/mlu_tools.h"
#include "cnrt.h"
#include "cnnl.h"
#include <stdio.h>

namespace oneflow {

typedef struct Prelu_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlTensorDescriptor_t alpha_desc = nullptr;
} Prelu;

template<DeviceType device_type, CamDataType T>
class PReluKernelCambricon final : public user_op::OpKernel {
 public:
  PReluKernelCambricon() = default;
  ~PReluKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    int ndim = x->shape().NumAxes();
    CHECK_EQ(ndim, 4);
    int n = x->shape().At(0);
    int h = x->shape().At(1);
    int w = x->shape().At(2);
    int c = x->shape().At(3);
    CHECK_EQ(y->shape(), x->shape());
    CHECK_EQ(alpha->shape().NumAxes(), ndim - 1);
    int an = 1;
    int ah = alpha->shape().At(0);
    int aw = alpha->shape().At(1);
    int ac = alpha->shape().At(2);
    // through test, we found that cnnlPrelu only support that
    // alpha dim_h and dim_w must be equal to 1
    CHECK_EQ(ah, 1);
    CHECK_EQ(aw, 1);
    CHECK(ac == 1 || ac == c);

    Prelu prelu;
    preluDataType datainfo;
    datainfo.input_dtype = convert(T);
    datainfo.output_dtype = convert(T);
    datainfo.alpha_dtype = convert(T);
    datainfo.layout = CNNL_LAYOUT_NHWC;

    set_tensor_desc(prelu.input_desc, n, h, w, c, datainfo.input_dtype, datainfo.layout);
    set_tensor_desc(prelu.alpha_desc, an, ah, aw, ac, datainfo.alpha_dtype, datainfo.layout);
    set_tensor_desc(prelu.output_desc, n, h, w, c, datainfo.output_dtype, datainfo.layout);

    CNNL_CHECK(cnnlPrelu(ctx->device_ctx()->cambricon_handle(), prelu.input_desc, x->dptr(),
                         prelu.alpha_desc, alpha->dptr(), prelu.output_desc, y->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PRELU_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("prelu")                           \
      .SetCreateFn<PReluKernelCambricon<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));

#ifdef WITH_CAMBRICON
REGISTER_PRELU_KERNEL(DeviceType::kCambricon, CamDataType::kFLOAT32)
#endif

}  // namespace oneflow

#endif
