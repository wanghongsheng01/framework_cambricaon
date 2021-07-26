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

typedef struct Convolution_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t weight_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlTensorDescriptor_t bias_desc = nullptr;
  cnnlConvolutionDescriptor_t conv_desc = nullptr;
  cnnlConvolutionForwardAlgo_t algo;
  void* workspace = nullptr;
  size_t workspace_size = 0;
} Convolution;

template<DeviceType device_type, CamDataType T>
class Conv2dKernelCambricon final : public user_op::OpKernel {
 public:
  Conv2dKernelCambricon() = default;
  ~Conv2dKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding_before");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const int& groups = ctx->Attr<int32_t>("groups");
    const int& filters = ctx->Attr<int32_t>("filters");
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    void* in_ptr = (void*)in->dptr();
    void* weight_ptr = (void*)weight->dptr();
    void* out_ptr = (void*)out->dptr();

    CHECK_EQ(in->shape().NumAxes(), 4)
        << "The number of axes of conv2d op input shape should equal to 4!";
    CHECK_EQ(weight->shape().NumAxes(), 4)
        << "The number of axes of conv2d op weight shape should equal to 4!";

    Convolution conv;
    // bool has_bias = false;
    convDataType datainfo;
    datainfo.input_dtype = convert(CamDataType::kINT8);
    datainfo.weight_dtype = convert(CamDataType::kINT8);
    datainfo.output_dtype = convert(T);

    int kh, kw, sh, sw, dh, dw;
    kh = kernel_size[0];
    kw = kernel_size[0];
    if (kernel_size.size() > 0) { kw = kernel_size[1]; }
    sh = strides[0];
    sw = strides[0];
    if (strides.size() > 0) { sw = strides[1]; }
    dh = dilation_rate[0];
    dw = dilation_rate[0];
    if (dilation_rate.size() > 0) { dw = dilation_rate[1]; }

    int pad_t = padding[0];
    int pad_b = padding[0];
    int pad_l = padding[1];
    int pad_r = padding[1];

    int pad[4] = {pad_t, pad_b, pad_l, pad_r};
    int stride[2] = {sh, sw};
    int dilation[2] = {dh, dw};
    int group_count = groups;

    set_tensor_desc(conv.input_desc, in->shape().At(0), in->shape().At(1), in->shape().At(2),
                    in->shape().At(3), datainfo.input_dtype, datainfo.layout);
    set_tensor_desc(conv.weight_desc, weight->shape().At(0), weight->shape().At(1),
                    weight->shape().At(2), weight->shape().At(3), datainfo.weight_dtype,
                    datainfo.layout);
    set_tensor_desc(conv.output_desc, out->shape().At(0), out->shape().At(1), out->shape().At(2),
                    out->shape().At(3), datainfo.output_dtype, datainfo.layout);
    set_tensor_desc(conv.bias_desc, out->shape().At(0), out->shape().At(1), out->shape().At(2),
                    out->shape().At(3), datainfo.output_dtype, datainfo.layout);
    // cast input data
    void* in_cast;
    int in_size = in->shape().At(0) * in->shape().At(1) * in->shape().At(2) * in->shape().At(3);
    CNRT_CHECK(cnrtMalloc(&(in_cast), in_size));
    CNRT_CHECK(cnrtMemset(in_cast, 0, in_size));

    void* in_pos = nullptr;
    void* in_scale = nullptr;
    CNRT_CHECK(cnrtMalloc((void**)&in_pos, sizeof(int32_t)));
    CNRT_CHECK(cnrtMalloc((void**)&in_scale, sizeof(float)));
    size_t in_workspace_size = 0;
    void* in_workspace = nullptr;
    cnnlTensorDescriptor_t in_desc = nullptr;
    set_tensor_desc(in_desc, in->shape().At(0), in->shape().At(1), in->shape().At(2),
                    in->shape().At(3), CNNL_DTYPE_FLOAT, datainfo.layout);
    CNNL_CHECK(cnnlGetQuantifyOnlineWorkspaceSize(ctx->device_ctx()->cambricon_handle(), in_desc,
                                                  &in_workspace_size));
    if (in_workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc((void**)&in_workspace, in_workspace_size));
      CNRT_CHECK(cnrtMemset(in_workspace, 0, in_workspace_size));
    }
    cnnlQuantifyOnline(ctx->device_ctx()->cambricon_handle(), false, in_desc, in_ptr, in_workspace,
                       in_workspace_size, in_pos, in_scale, conv.input_desc, in_cast);
    ctx->device_ctx()->SyncDevice();
    int pos1 = 0;
    CNRT_CHECK(cnrtMemcpy(&pos1, in_pos, sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST));
    float scale1 = 0;
    CNRT_CHECK(cnrtMemcpy(&scale1, in_scale, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));

    // cast weight data
    void* weight_cast;
    int weight_size = weight->shape().At(0) * weight->shape().At(1) * weight->shape().At(2)
                      * weight->shape().At(3);
    CNRT_CHECK(cnrtMalloc(&(weight_cast), weight_size));
    CNRT_CHECK(cnrtMemset(weight_cast, 0, weight_size));
    void* weight_pos = nullptr;
    void* weight_scale = nullptr;
    CNRT_CHECK(cnrtMalloc((void**)&weight_pos, sizeof(int32_t)));
    CNRT_CHECK(cnrtMalloc((void**)&weight_scale, sizeof(float)));
    size_t weight_workspace_size = 0;
    void* weight_workspace = nullptr;
    cnnlTensorDescriptor_t weight_desc = nullptr;
    set_tensor_desc(weight_desc, weight->shape().At(0), weight->shape().At(1),
                    weight->shape().At(2), weight->shape().At(3), CNNL_DTYPE_FLOAT,
                    datainfo.layout);
    CNNL_CHECK(cnnlGetQuantifyOnlineWorkspaceSize(ctx->device_ctx()->cambricon_handle(),
                                                  weight_desc, &weight_workspace_size));
    if (weight_workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc((void**)&weight_workspace, weight_workspace_size));
      CNRT_CHECK(cnrtMemset(weight_workspace, 0, weight_workspace_size));
    }
    cnnlQuantifyOnline(ctx->device_ctx()->cambricon_handle(), false, weight_desc, weight_ptr,
                       weight_workspace, weight_workspace_size, weight_pos, weight_scale,
                       conv.weight_desc, weight_cast);
    ctx->device_ctx()->SyncDevice();
    int pos2 = 0;
    CNRT_CHECK(cnrtMemcpy(&pos2, weight_pos, sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST));
    float scale2 = 0;
    CNRT_CHECK(cnrtMemcpy(&scale2, weight_scale, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));

    void* bias_cast;
    int bias_size =
        out->shape().At(0) * out->shape().At(1) * out->shape().At(2) * out->shape().At(3);
    CNRT_CHECK(cnrtMalloc(&(bias_cast), bias_size));
    CNRT_CHECK(cnrtMemset(bias_cast, 0, bias_size));

    CNNL_CHECK(cnnlCreateConvolutionDescriptor(&conv.conv_desc));
    conv.algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
    CNNL_CHECK(cnnlSetConvolutionDescriptor(conv.conv_desc, 4, pad, stride, dilation, group_count));
    CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
        ctx->device_ctx()->cambricon_handle(), conv.input_desc, conv.weight_desc, conv.output_desc,
        conv.conv_desc, conv.algo, &(conv.workspace_size)));

    if (conv.workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc(&(conv.workspace), conv.workspace_size));
      CNRT_CHECK(cnrtMemset(conv.workspace, 0, conv.workspace_size));
    }

    cnnlSetTensorDescriptorPositionAndScale(conv.input_desc, pos1, scale1);
    cnnlSetTensorDescriptorPositionAndScale(conv.weight_desc, pos2, scale2);
    CNNL_CHECK(cnnlConvolutionForward(
        ctx->device_ctx()->cambricon_handle(), conv.conv_desc, conv.algo, nullptr, conv.input_desc,
        in_cast, conv.weight_desc, weight_cast, conv.bias_desc, bias_cast, conv.workspace,
        conv.workspace_size, nullptr, conv.output_desc, out_ptr));
    ctx->device_ctx()->SyncDevice();

    if (conv.workspace != nullptr) { CNRT_CHECK(cnrtFree(conv.workspace)); }
    if (in_workspace != nullptr) { CNRT_CHECK(cnrtFree(in_workspace)); }
    if (weight_workspace != nullptr) { CNRT_CHECK(cnrtFree(weight_workspace)); }
    CNRT_CHECK(cnrtFree(in_pos));
    CNRT_CHECK(cnrtFree(in_scale));
    CNRT_CHECK(cnrtFree(weight_pos));
    CNRT_CHECK(cnrtFree(weight_scale));
    CNRT_CHECK(cnrtFree(in_cast));
    CNRT_CHECK(cnrtFree(weight_cast));
    CNRT_CHECK(cnrtFree(bias_cast));
    CNNL_CHECK(cnnlDestroyConvolutionDescriptor(conv.conv_desc));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONV2D_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("conv2d")                           \
      .SetCreateFn<Conv2dKernelCambricon<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));

#ifdef WITH_CAMBRICON
REGISTER_CONV2D_KERNEL(DeviceType::kCambricon, CamDataType::kFLOAT32)
#endif

}  // namespace

}  // namespace oneflow
#endif
