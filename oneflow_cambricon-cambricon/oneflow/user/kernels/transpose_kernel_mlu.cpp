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

namespace oneflow {

namespace user_op {

struct Transpose {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlTransposeDescriptor_t transpose_desc = nullptr;
  void* input_data;
  void* output_data;
  float hw_time = 0;
};

template<DeviceType device_type>
class TransposeKernel final : public user_op::OpKernel {
 public:
  TransposeKernel() = default;
  ~TransposeKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("input", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto& perm = ctx->Attr<std::vector<int32_t>>("perm");

    CHECK_EQ(tensor_in->shape().NumAxes(), 4) << "4 dims supported only";
    CHECK_EQ(tensor_out->shape().NumAxes(), 4) << "4 dims supported only";
    CHECK_EQ(perm.size(), 4) << "4 dims supported only";

    Transpose trans;
    TransposeType datainfo;
    datainfo.input_dtype = convert(CamDataType::kFLOAT32);
    datainfo.output_dtype = convert(CamDataType::kFLOAT32);

    int64_t in_shape[4] = {tensor_in->shape().At(0), tensor_in->shape().At(1),
                           tensor_in->shape().At(2), tensor_in->shape().At(3)};
    int64_t out_shape[4] = {in_shape[perm[0]], in_shape[perm[1]], in_shape[perm[2]],
                            in_shape[perm[3]]};

    set_tensor_desc(trans.input_desc, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    datainfo.input_dtype, datainfo.layout);
    set_tensor_desc(trans.output_desc, out_shape[0], out_shape[1], out_shape[2], out_shape[3],
                    datainfo.output_dtype, datainfo.layout);

    CNNL_CHECK(cnnlCreateTransposeDescriptor(&trans.transpose_desc));
    CNNL_CHECK(cnnlSetTransposeDescriptor(trans.transpose_desc, 4, &perm[0]));

    CNNL_CHECK(cnnlTranspose(ctx->device_ctx()->cambricon_handle(), trans.transpose_desc,
                             trans.input_desc, (const void*)tensor_in->dptr(), trans.output_desc,
                             (void*)tensor_out->dptr()));

    CNNL_CHECK(cnnlDestroyTransposeDescriptor(trans.transpose_desc));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TRANSPOSE_KERNEL(device, dtype)                     \
  REGISTER_USER_KERNEL("transpose")                                  \
      .SetCreateFn<TransposeKernel<device>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)           \
                       & (user_op::HobDataType("input", 0) == dtype) \
                       & (user_op::HobDataType("output", 0) == dtype));

REGISTER_TRANSPOSE_KERNEL(DeviceType::kCambricon, DataType::kFloat);

}  // namespace user_op

}  // namespace oneflow

#endif  // WITH_CAMBRICON
