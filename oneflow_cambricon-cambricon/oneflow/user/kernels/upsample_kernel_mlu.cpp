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

namespace oneflow {

namespace user_op {

typedef struct Interp {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  void* input_data;
  void* output_data;
  float hw_time = 0;
};

template<DeviceType device_type>
class UpsampleMLUKernel final : public user_op::OpKernel {
 public:
  UpsampleMLUKernel() = default;
  ~UpsampleMLUKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("y", 0);
    CHECK_EQ(tensor_in->shape().NumAxes(), 4) << "4 dims supported only(n,h,w,c)";
    CHECK_EQ(tensor_out->shape().NumAxes(), 4) << "4 dims supported only(n,h,w,c)";
    CHECK_EQ(ctx->Attr<std::string>("data_format"), "channels_last");
    const std::string interpolation = ctx->Attr<std::string>("interpolation");
    cnnlInterpMode_t mode = interpolation == "nearest" ? CNNL_INTERP_NEAREST : CNNL_INTERP_BILINEAR;
    const bool align_center = interpolation == "nearest" ? false : true;
    const bool align_corners = !align_center;

    Interp interp;
    InterpType datainfo;
    datainfo.input_dtype = convert(CamDataType::kFLOAT32);
    datainfo.output_dtype = convert(CamDataType::kFLOAT32);
    set_tensor_desc(interp.input_desc, tensor_in->shape().At(0), tensor_in->shape().At(1),
                    tensor_in->shape().At(2), tensor_in->shape().At(3), datainfo.input_dtype,
                    datainfo.layout);
    set_tensor_desc(interp.output_desc, tensor_out->shape().At(0), tensor_out->shape().At(1),
                    tensor_out->shape().At(2), tensor_out->shape().At(3), datainfo.output_dtype,
                    datainfo.layout);

    CNNL_CHECK(cnnlInterp(ctx->device_ctx()->cambricon_handle(), align_corners, align_center, mode,
                          interp.input_desc, (void*)tensor_in->dptr(), interp.output_desc,
                          (void*)tensor_out->dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_KERNEL(device, dtype)                  \
  REGISTER_USER_KERNEL("upsample")                               \
      .SetCreateFn<UpsampleMLUKernel<device>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)       \
                       & (user_op::HobDataType("x", 0) == dtype) \
                       & (user_op::HobDataType("y", 0) == dtype));

REGISTER_UPSAMPLE_KERNEL(DeviceType::kCambricon, DataType::kFloat);

}  // namespace user_op

}  // namespace oneflow

#endif  // WITH_CAMBRICON
