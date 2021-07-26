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

typedef struct Matmul_ {
  cnnlTensorDescriptor_t a_desc = nullptr;
  cnnlTensorDescriptor_t b_desc = nullptr;
  cnnlTensorDescriptor_t c_desc = nullptr;
} Matmul;

template<DeviceType device_type, CamDataType T>
class MatmulKernelCambricon final : public user_op::OpKernel {
 public:
  MatmulKernelCambricon() = default;
  ~MatmulKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");
    user_op::Tensor* c = ctx->Tensor4ArgNameAndIndex("out", 0);

    void* a_ptr = (void*)a->dptr();
    void* b_ptr = (void*)b->dptr();
    void* c_ptr = (void*)c->dptr();

    Matmul matmul;
    MatMulType datainfo;
    datainfo.input_dtype = convert(CamDataType::kINT8);
    datainfo.output_dtype = convert(T);

    CHECK_EQ(a->shape().NumAxes(), 2);
    CHECK_EQ(b->shape().NumAxes(), 2);

    set_tensor_desc_matmul(matmul.a_desc, a->shape().At(0), a->shape().At(1), CNNL_DTYPE_INT8,
                           datainfo.layout);

    set_tensor_desc_matmul(matmul.b_desc, b->shape().At(0), b->shape().At(1), CNNL_DTYPE_INT8,
                           datainfo.layout);

    set_tensor_desc_matmul(matmul.c_desc, c->shape().At(0), c->shape().At(1), datainfo.output_dtype,
                           datainfo.layout);
    // cast a
    void* a_cast;
    int a_size = a->shape().elem_cnt();
    CNRT_CHECK(cnrtMalloc(&(a_cast), a_size));
    CNRT_CHECK(cnrtMemset(a_cast, 0, a_size));

    void* a_pos = nullptr;
    void* a_scale = nullptr;
    CNRT_CHECK(cnrtMalloc((void**)&a_pos, sizeof(int32_t)));
    CNRT_CHECK(cnrtMalloc((void**)&a_scale, sizeof(float)));
    size_t a_workspace_size = 0;
    void* a_workspace = nullptr;
    cnnlTensorDescriptor_t a_desc = nullptr;
    set_tensor_desc_matmul(a_desc, a->shape().At(0), a->shape().At(1), CNNL_DTYPE_FLOAT,
                           datainfo.layout);

    CNNL_CHECK(cnnlGetQuantifyOnlineWorkspaceSize(ctx->device_ctx()->cambricon_handle(), a_desc,
                                                  &a_workspace_size));
    if (a_workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc((void**)&a_workspace, a_workspace_size));
      CNRT_CHECK(cnrtMemset(a_workspace, 0, a_workspace_size));
    }
    cnnlQuantifyOnline(ctx->device_ctx()->cambricon_handle(), false, a_desc, a_ptr, a_workspace,
                       a_workspace_size, a_pos, a_scale, matmul.a_desc, a_cast);
    ctx->device_ctx()->SyncDevice();
    int a_pos_ = 0;
    float a_scale_ = 0;
    CNRT_CHECK(cnrtMemcpy(&a_pos_, a_pos, sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(&a_scale_, a_scale, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnnlSetTensorDescriptorPositionAndScale(matmul.a_desc, a_pos_, a_scale_);

    // cast b
    void* b_cast;
    int b_size = b->shape().elem_cnt();
    CNRT_CHECK(cnrtMalloc(&(b_cast), b_size));

    void* b_pos = nullptr;
    void* b_scale = nullptr;
    CNRT_CHECK(cnrtMalloc((void**)&b_pos, sizeof(int32_t)));
    CNRT_CHECK(cnrtMalloc((void**)&b_scale, sizeof(float)));
    size_t b_workspace_size = 0;
    void* b_workspace = nullptr;
    cnnlTensorDescriptor_t b_desc = nullptr;
    set_tensor_desc_matmul(b_desc, b->shape().At(0), b->shape().At(1), CNNL_DTYPE_FLOAT,
                           datainfo.layout);

    CNNL_CHECK(cnnlGetQuantifyOnlineWorkspaceSize(ctx->device_ctx()->cambricon_handle(), b_desc,
                                                  &b_workspace_size));
    if (b_workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc((void**)&b_workspace, b_workspace_size));
      CNRT_CHECK(cnrtMemset(b_workspace, 0, b_workspace_size));
    }
    cnnlQuantifyOnline(ctx->device_ctx()->cambricon_handle(), false, b_desc, b_ptr, b_workspace,
                       b_workspace_size, b_pos, b_scale, matmul.b_desc, b_cast);
    ctx->device_ctx()->SyncDevice();
    int b_pos_ = 0;
    float b_scale_ = 0;
    CNRT_CHECK(cnrtMemcpy(&b_pos_, b_pos, sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(&b_scale_, b_scale, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnnlSetTensorDescriptorPositionAndScale(matmul.b_desc, b_pos_, b_scale_);

    bool is_trans_a = transpose_a;
    bool is_trans_b = transpose_b;
    float* alpha = (float*)malloc(1 * sizeof(float));
    alpha[0] = 1.0;
    float* beta = (float*)malloc(1 * sizeof(float));
    beta[0] = 0.0;

    CNNL_CHECK(cnnlMatMul(ctx->device_ctx()->cambricon_handle(), is_trans_a, is_trans_b,
                          (void*)alpha, matmul.a_desc, a_cast, matmul.b_desc, b_cast, (void*)beta,
                          matmul.c_desc, c_ptr));

    if (a_workspace != nullptr) { CNRT_CHECK(cnrtFree(a_workspace)); }
    if (b_workspace != nullptr) { CNRT_CHECK(cnrtFree(b_workspace)); }
    CNRT_CHECK(cnrtFree(a_pos));
    CNRT_CHECK(cnrtFree(a_scale));
    CNRT_CHECK(cnrtFree(b_pos));
    CNRT_CHECK(cnrtFree(b_scale));
    CNRT_CHECK(cnrtFree(a_cast));
    CNRT_CHECK(cnrtFree(b_cast));
    free(alpha);
    free(beta);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATMUL_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("matmul")                           \
      .SetCreateFn<MatmulKernelCambricon<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));

REGISTER_MATMUL_KERNEL(DeviceType::kCambricon, CamDataType::kFLOAT32)

}  // namespace oneflow

#endif
