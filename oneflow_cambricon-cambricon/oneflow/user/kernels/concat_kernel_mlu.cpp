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
struct Concat {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  void* workspace = nullptr;
  size_t workspace_size = 0;
  float hw_time = 0;
};

template<DeviceType device_type>
class ConcatKernel final : public user_op::OpKernel {
 public:
  ConcatKernel() = default;
  ~ConcatKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    int concat_num = ctx->inputs().size();

    ConcatType datainfo;
    datainfo.input_dtype = convert(CamDataType::kFLOAT32);
    datainfo.output_dtype = convert(CamDataType::kFLOAT32);

    Concat concat;
    set_tensor_desc(concat.input_desc, out_tensor->shape().At(0), out_tensor->shape().At(1),
                    out_tensor->shape().At(2), out_tensor->shape().At(3), datainfo.input_dtype,
                    datainfo.layout);

    cnnlTensorDescriptor_t* input_descs = new cnnlTensorDescriptor_t[concat_num];
    void** in_dptrs = new void*[concat_num];

    for (int32_t i = 0; i < concat_num; ++i) {
      Tensor* in_tensor_i = ctx->Tensor4ArgNameAndIndex("in", i);
      // TODO(yaochi): check below may only work well in 1node 1device env.
      CHECK_EQ(in_tensor_i->shape().NumAxes(), 4) << "Dimensions of tensor could only be 4 by now";
      CHECK_EQ(in_tensor_i->shape().NumAxes(), out_tensor->shape().NumAxes())
          << "Dimensions of in and out tensors should be same";

      in_dptrs[i] = (void*)in_tensor_i->dptr();
      set_tensor_desc(concat.input_desc, in_tensor_i->shape().At(0), in_tensor_i->shape().At(1),
                      in_tensor_i->shape().At(2), in_tensor_i->shape().At(3), datainfo.input_dtype,
                      datainfo.layout);
      input_descs[i] = concat.input_desc;
    }

    set_tensor_desc(concat.output_desc, out_tensor->shape().At(0), out_tensor->shape().At(1),
                    out_tensor->shape().At(2), out_tensor->shape().At(3), datainfo.output_dtype,
                    datainfo.layout);

    CNNL_CHECK(cnnlGetConcatWorkspaceSize(ctx->device_ctx()->cambricon_handle(), concat_num,
                                          &(concat.workspace_size)));

    if (concat.workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc(&(concat.workspace), concat.workspace_size));
      CNRT_CHECK(cnrtMemsetD8(concat.workspace, 0, concat.workspace_size));
    }

    void* out_ptr = (void*)out_tensor->dptr();
    CNNL_CHECK(cnnlConcat(ctx->device_ctx()->cambricon_handle(), concat_num, axis, input_descs,
                          in_dptrs, concat.workspace, concat.workspace_size, concat.output_desc,
                          out_ptr));

    if (concat.workspace) {
      CNRT_CHECK(cnrtFree(concat.workspace));
      concat.workspace = nullptr;
    }

    if (in_dptrs) {
      delete[] in_dptrs;
      in_dptrs = nullptr;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONCAT_KERNEL(device, dtype)                                         \
  REGISTER_USER_KERNEL("concat").SetCreateFn<ConcatKernel<device>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device) & (user_op::HobDataType("in", 0) == dtype)  \
      & (user_op::HobDataType("out", 0) == dtype));

REGISTER_CONCAT_KERNEL(DeviceType::kCambricon, DataType::kFloat);

}  // namespace user_op

}  // namespace oneflow

#endif  // WITH_CAMBRICON
