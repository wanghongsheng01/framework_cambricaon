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
#include "oneflow/core/kernel/mlu_tools.h"
#include "cnrt.h"
#include "cnnl.h"
#include <stdio.h>

namespace oneflow {

typedef struct Add_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
} Add;

template<DeviceType device_type, CamDataType T>
class AddKernelCambricon : public user_op::OpKernel {
 public:
  AddKernelCambricon() = default;
  ~AddKernelCambricon() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    size_t in_num = ctx->inputs().size();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    void* out_dptr = static_cast<void*>(out->mut_dptr());

    std::vector<const void*> input_dptrs_vec(in_num);
    const auto& first_input_shape = ctx->Tensor4ArgNameAndIndex("in", 0)->shape();
    for (size_t i = 0; i < input_dptrs_vec.size(); ++i) {
      const auto* input_i_tensor = ctx->Tensor4ArgNameAndIndex("in", i);
      CHECK_EQ(first_input_shape, input_i_tensor->shape());
      input_dptrs_vec[i] = input_i_tensor->dptr();
    }
    size_t ndim = first_input_shape.NumAxes();
    std::vector<int> dim_vec(ndim);
    for (size_t i = 0; i < ndim; ++i) { dim_vec[i] = first_input_shape.At(i); }

    Add add;
    AddType datainfo;
    datainfo.input_dtype = convert(T);
    datainfo.output_dtype = convert(T);
    set_tensor_desc(add.input_desc, dim_vec.size(), dim_vec.data(), datainfo.input_dtype,
                    datainfo.layout);
    set_tensor_desc(add.output_desc, dim_vec.size(), dim_vec.data(), datainfo.output_dtype,
                    datainfo.layout);
    std::vector<cnnlTensorDescriptor_t> input_descs_vec{in_num, add.input_desc};
    CNNL_CHECK(cnnlAddN(ctx->device_ctx()->cambricon_handle(), input_descs_vec.data(),
                        input_dptrs_vec.data(), in_num, add.output_desc, out_dptr));
  }
};

#define REGISTER_ADDN_CAMB_KERNEL(dtype)                                \
  REGISTER_USER_KERNEL("add_n")                                         \
      .SetCreateFn<AddKernelCambricon<DeviceType::kCambricon, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCambricon));

REGISTER_ADDN_CAMB_KERNEL(CamDataType::kFLOAT32)

}  // namespace oneflow

#endif  // WITH_CAMBRICON
