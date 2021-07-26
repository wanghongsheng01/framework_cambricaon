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

typedef struct Instancenorm_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlTensorDescriptor_t scale_bias_desc = nullptr;
  cnnlTensorDescriptor_t mean_var_desc = nullptr;
} Instancenorm;

struct InstanceNormType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlDataType_t scale_bias_desc_dtype;
  cnnlDataType_t mean_var_desc_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

template<DeviceType device_type>
class InstanceNorm2DKernelCambricon final : public user_op::OpKernel {
 public:
  InstanceNorm2DKernelCambricon() = default;
  ~InstanceNorm2DKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* var = ctx->Tensor4ArgNameAndIndex("var", 0);
    const float eps = ctx->Attr<float>("eps");

    Instancenorm instance_norm;
    InstanceNormType datainfo;
    datainfo.input_dtype = convert(CamDataType::kFLOAT32);
    datainfo.output_dtype = convert(CamDataType::kFLOAT32);
    datainfo.scale_bias_desc_dtype = convert(CamDataType::kFLOAT32);
    datainfo.mean_var_desc_dtype = convert(CamDataType::kFLOAT32);

    set_tensor_desc_3d(instance_norm.input_desc, in->shape().At(0),
                       in->shape().At(1) * in->shape().At(2), in->shape().At(3),
                       datainfo.input_dtype, datainfo.layout);
    set_tensor_desc_3d(instance_norm.mean_var_desc, var->shape().At(0), var->shape().At(1),
                       var->shape().At(2), datainfo.mean_var_desc_dtype, datainfo.layout);

    void* in_ptr = (void*)in->dptr();

    // calculate variance
    void* var_ptr = (void*)var->dptr();
    CNNL_CHECK(cnnlStdForward(ctx->device_ctx()->cambricon_handle(), 1, false,
                              instance_norm.input_desc, in_ptr, instance_norm.mean_var_desc,
                              var_ptr));
    CNNL_CHECK(cnnlSquare(ctx->device_ctx()->cambricon_handle(), instance_norm.mean_var_desc,
                          var_ptr, instance_norm.mean_var_desc, var_ptr));

    // calculate mean
    void* mean_ptr = (void*)mean->dptr();
    cnnlReduceDescriptor_t reduce_desc;
    CNNL_CHECK(cnnlCreateReduceDescriptor(&reduce_desc));

    int axis[1];
    axis[0] = 1;
    CNNL_CHECK(cnnlSetReduceDescriptor(reduce_desc, axis, 1, CNNL_REDUCE_AVG, datainfo.input_dtype,
                                       CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_FLATTENED_INDICES,
                                       CNNL_32BIT_INDICES));

    CNNL_CHECK(cnnlReduce(ctx->device_ctx()->cambricon_handle(), reduce_desc, nullptr, 0, nullptr,
                          instance_norm.input_desc, in_ptr, 0, nullptr, nullptr,
                          instance_norm.mean_var_desc, mean_ptr));

    CNNL_CHECK(cnnlDestroyReduceDescriptor(reduce_desc));

    // calculate instance norm
    CNNL_CHECK(cnnlDestroyTensorDescriptor(instance_norm.input_desc));
    set_tensor_desc(instance_norm.input_desc, in->shape().At(0), in->shape().At(1),
                    in->shape().At(2), in->shape().At(3), datainfo.input_dtype, datainfo.layout);
    CNNL_CHECK(cnnlDestroyTensorDescriptor(instance_norm.mean_var_desc));
    set_tensor_desc(instance_norm.mean_var_desc, in->shape().At(0), 1, 1, in->shape().At(3),
                    datainfo.mean_var_desc_dtype, datainfo.layout);

    set_tensor_desc(instance_norm.output_desc, out->shape().At(0), out->shape().At(1),
                    out->shape().At(2), out->shape().At(3), datainfo.output_dtype, datainfo.layout);
    set_tensor_desc(instance_norm.scale_bias_desc, 1, 1, 1, in->shape().At(3),
                    datainfo.scale_bias_desc_dtype, datainfo.layout);

    void* gamma_ptr = (void*)gamma->dptr();
    void* beta_ptr = (void*)beta->dptr();
    void* out_ptr = (void*)out->dptr();
    CNNL_CHECK(cnnlInstanceNormInference(
        ctx->device_ctx()->cambricon_handle(), instance_norm.input_desc, in_ptr,
        instance_norm.mean_var_desc, mean_ptr, var_ptr, instance_norm.scale_bias_desc, gamma_ptr,
        beta_ptr, eps, instance_norm.output_desc, out_ptr));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_KERNEL(device, dtype)                 \
  REGISTER_USER_KERNEL("instance_norm_2d")                  \
      .SetCreateFn<InstanceNorm2DKernelCambricon<device>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)  \
                       & (user_op::HobDataType("in", 0) == dtype))

#ifdef WITH_CAMBRICON
REGISTER_RELU_KERNEL(DeviceType::kCambricon, DataType::kFloat);
#endif

}  // namespace

}  // namespace oneflow

#endif
