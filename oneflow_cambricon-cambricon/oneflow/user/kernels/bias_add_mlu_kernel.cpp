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

typedef struct Bias_ {
  cnnlTensorDescriptor_t a_desc = nullptr;
  cnnlTensorDescriptor_t b_desc = nullptr;
} Bias;

template<DeviceType device_type, CamDataType T>
class BiasAddKernelCambricon final : public user_op::OpKernel {
 public:
  BiasAddKernelCambricon() = default;
  ~BiasAddKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    int n = 0;
    int h = 0;
    int w = 0;
    int c = 0;
    int adim = a_tensor->shape().NumAxes();
    int bdim = b_tensor->shape().NumAxes();
    CHECK_EQ(bdim, 1);
    if (adim == 4) {
      n = a_tensor->shape().At(0);
      h = a_tensor->shape().At(1);
      w = a_tensor->shape().At(2);
      c = a_tensor->shape().At(3);
    } else if (adim == 2) {
      n = a_tensor->shape().At(0);
      h = 1;
      w = 1;
      c = a_tensor->shape().At(1);
    } else {
      UNIMPLEMENTED();
    }

    Bias bias;
    void* a_ptr = (void*)a_tensor->dptr();
    void* b_ptr = (void*)b_tensor->dptr();
    void* out_ptr = (void*)out_tensor->dptr();
    BiasAddType datainfo;
    datainfo.a_dtype = convert(T);
    datainfo.b_dtype = convert(T);
    datainfo.layout = CNNL_LAYOUT_NHWC;

    float alpha = 1;
    float beta = 1;

    set_tensor_desc(bias.a_desc, n, h, w, c, datainfo.a_dtype, datainfo.layout);
    set_tensor_desc_biasadd(bias.b_desc, c, datainfo.b_dtype, datainfo.layout);
    size_t workspace_size = 0;
    void* workspace = nullptr;
    CNNL_CHECK(cnnlGetBiasAddWorkspaceSize(ctx->device_ctx()->cambricon_handle(), bias.b_desc,
                                           bias.a_desc, &workspace_size));
    if (workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc((void**)&workspace, workspace_size));
      CNRT_CHECK(cnrtMemset(workspace, 0, workspace_size));
    }
    ctx->device_ctx()->SyncDevice();
    CNRT_CHECK(cnrtMemcpy(out_ptr, a_ptr, n * h * w * c * 4, CNRT_MEM_TRANS_DIR_NODIR));

    CNNL_CHECK(cnnlBiasAdd(ctx->device_ctx()->cambricon_handle(), &alpha, bias.b_desc, b_ptr,
                           workspace, workspace_size, &beta, bias.a_desc, out_ptr));
    if (workspace != nullptr) { CNRT_CHECK(cnrtFree(workspace)); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BIASADD_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("bias_add")                          \
      .SetCreateFn<BiasAddKernelCambricon<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));

#ifdef WITH_CAMBRICON
REGISTER_BIASADD_KERNEL(DeviceType::kCambricon, CamDataType::kFLOAT32)
#endif

}  // namespace oneflow

#endif
