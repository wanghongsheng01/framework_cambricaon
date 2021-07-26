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

typedef struct Pooling_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlPoolingDescriptor_t pool_desc = nullptr;
  void* workspace = nullptr;
  size_t workspace_size = 0;
} Pooling;

template<DeviceType device_type, CamDataType T>
class MaxPoolKernelCambricon : public user_op::OpKernel {
 public:
  MaxPoolKernelCambricon() = default;
  ~MaxPoolKernelCambricon() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::string& padding = ctx->Attr<std::string>("padding");
    const std::vector<int32_t>& pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    CHECK_EQ(ceil_mode, false) << "The ceil_mode of maxpool op should equal to 0!";

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    void* x_ptr = (void*)x->dptr();
    void* y_ptr = (void*)y->dptr();

    CHECK_EQ(x->shape().NumAxes(), 4)
        << "The number of axes of maxpool op input shape should equal to 4!";

    cnnlPoolingMode_t model_type = CNNL_POOLING_MAX;
    cnnlNanPropagation_t maxpooling_nan_opt = CNNL_NOT_PROPAGATE_NAN;
    Pooling pool;
    poolDataType datainfo;
    datainfo.input_dtype = convert(T);
    datainfo.output_dtype = convert(T);
    set_tensor_desc(pool.input_desc, x->shape().At(0), x->shape().At(1), x->shape().At(2),
                    x->shape().At(3), datainfo.input_dtype, datainfo.layout);
    set_tensor_desc(pool.output_desc, y->shape().At(0), y->shape().At(1), y->shape().At(2),
                    y->shape().At(3), datainfo.output_dtype, datainfo.layout);
    CNNL_CHECK(cnnlCreatePoolingDescriptor(&pool.pool_desc));
    int window_height = pool_size[0];
    int window_width = pool_size[0];
    if (pool_size.size() > 1) { window_width = pool_size[1]; }

    int vertical_stride = strides[0];
    int horizon_stride = strides[0];
    if (strides.size() > 1) { horizon_stride = strides[1]; }

    int pad_t = 0;
    int pad_b = 0;
    int pad_l = 0;
    int pad_r = 0;

    int input_height = x->shape().At(1);
    int output_height = (input_height + (vertical_stride - 1)) / vertical_stride;
    int total_vertical_padding =
        (output_height - 1) * vertical_stride + window_height - input_height;
    int input_width = x->shape().At(2);
    int output_width = (input_width + (horizon_stride - 1)) / horizon_stride;
    int total_horizon_padding = (output_width - 1) * horizon_stride + window_width - input_width;

    int pad_b_t_small = total_vertical_padding / 2;
    int pad_b_t_big = total_vertical_padding - pad_b_t_small;
    int pad_l_r_small = total_horizon_padding / 2;
    int pad_l_r_big = total_horizon_padding - pad_l_r_small;

    if (padding == "same_lower") {
      pad_b = pad_b_t_small;
      pad_t = pad_b_t_big;
      pad_r = pad_l_r_small;
      pad_l = pad_l_r_big;
    } else if (padding == "same_upper") {
      pad_b = pad_b_t_big;
      pad_t = pad_b_t_small;
      pad_r = pad_l_r_big;
      pad_l = pad_l_r_small;
    } else if (padding == "valid") {
      pad_t = 0;
      pad_b = 0;
      pad_l = 0;
      pad_r = 0;
    } else {
      UNIMPLEMENTED();
    }

    int output_dim_h = (x->shape().At(1) + pad_t + pad_b - window_height) / vertical_stride + 1;
    int output_dim_w = (x->shape().At(2) + pad_l + pad_r - window_width) / horizon_stride + 1;

    CNNL_CHECK(cnnlCreatePoolingDescriptor(&pool.pool_desc));
    CNNL_CHECK(cnnlSetPooling2dDescriptor(pool.pool_desc, model_type, maxpooling_nan_opt,
                                          window_height, window_width, pad_t, pad_b, pad_l, pad_r,
                                          vertical_stride, horizon_stride));
    CNNL_CHECK(cnnlGetPoolingWorkspaceSize(ctx->device_ctx()->cambricon_handle(), model_type,
                                           output_dim_w, output_dim_h, &(pool.workspace_size)));
    if (pool.workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc(&(pool.workspace), pool.workspace_size));
      CNRT_CHECK(cnrtMemset(pool.workspace, 0, pool.workspace_size));
    }
    CNNL_CHECK(cnnlPoolingForward(ctx->device_ctx()->cambricon_handle(), pool.pool_desc, nullptr,
                                  pool.input_desc, x_ptr, nullptr, pool.output_desc, y_ptr,
                                  pool.workspace, pool.workspace_size));
    if (pool.workspace != nullptr) { CNRT_CHECK(cnrtFree(pool.workspace)); }
    CNNL_CHECK(cnnlDestroyPoolingDescriptor(pool.pool_desc));
  }
};

template<DeviceType device_type, CamDataType T>
class AvgPoolKernelCambricon : public user_op::OpKernel {
 public:
  AvgPoolKernelCambricon() = default;
  ~AvgPoolKernelCambricon() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::string& padding = ctx->Attr<std::string>("padding");
    const std::vector<int32_t>& pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
    const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    CHECK_EQ(ceil_mode, false) << "The ceil_mode of maxpool op should equal to 0!";

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    void* x_ptr = (void*)x->dptr();
    void* y_ptr = (void*)y->dptr();

    CHECK_EQ(x->shape().NumAxes(), 4)
        << "The number of axes of avg op input shape should equal to 4!";

    cnnlPoolingMode_t model_type = CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    cnnlNanPropagation_t maxpooling_nan_opt = CNNL_NOT_PROPAGATE_NAN;
    Pooling pool;
    poolDataType datainfo;
    datainfo.input_dtype = convert(T);
    datainfo.output_dtype = convert(T);
    set_tensor_desc(pool.input_desc, x->shape().At(0), x->shape().At(1), x->shape().At(2),
                    x->shape().At(3), datainfo.input_dtype, datainfo.layout);
    set_tensor_desc(pool.output_desc, y->shape().At(0), y->shape().At(1), y->shape().At(2),
                    y->shape().At(3), datainfo.output_dtype, datainfo.layout);
    CNNL_CHECK(cnnlCreatePoolingDescriptor(&pool.pool_desc));
    int window_height = pool_size[0];
    int window_width = pool_size[0];
    if (pool_size.size() > 1) { window_width = pool_size[1]; }

    int vertical_stride = strides[0];
    int horizon_stride = strides[0];
    if (strides.size() > 1) { horizon_stride = strides[1]; }

    int pad_t = 0;
    int pad_b = 0;
    int pad_l = 0;
    int pad_r = 0;

    if (padding == "same_lower") {
      int input_height = x->shape().At(1);
      int output_height = (input_height + (vertical_stride - 1))
                          / vertical_stride;  // a.k.a. ceil(1.0 * input_height / vertical_stride)
      int total_vertical_padding =
          (output_height - 1) * vertical_stride + window_height - input_height;
      pad_b = total_vertical_padding / 2;
      pad_t = total_vertical_padding - pad_b;

      int input_width = x->shape().At(2);
      int output_width = (input_width + (horizon_stride - 1)) / horizon_stride;
      int total_horizon_padding = (output_width - 1) * horizon_stride + window_width - input_width;
      pad_r = total_horizon_padding / 2;
      pad_l = total_horizon_padding - pad_l;
    } else if (padding == "same_upper") {
      int input_height = x->shape().At(1);
      int output_height = (input_height + (vertical_stride - 1))
                          / vertical_stride;  // a.k.a. ceil(1.0 * input_height / vertical_stride)
      int total_vertical_padding =
          (output_height - 1) * vertical_stride + window_height - input_height;
      pad_t = total_vertical_padding / 2;
      pad_b = total_vertical_padding - pad_t;

      int input_width = x->shape().At(2);
      int output_width = (input_width + (horizon_stride - 1)) / horizon_stride;
      int total_horizon_padding = (output_width - 1) * horizon_stride + window_width - input_width;
      pad_l = total_horizon_padding / 2;
      pad_r = total_horizon_padding - pad_l;
    } else if (padding == "valid") {
      pad_t = 0;
      pad_b = 0;
      pad_l = 0;
      pad_r = 0;
    } else {
      UNIMPLEMENTED();
    }

    int output_dim_h = (x->shape().At(1) + pad_t + pad_b - window_height) / vertical_stride + 1;
    int output_dim_w = (x->shape().At(2) + pad_l + pad_r - window_width) / horizon_stride + 1;

    CNNL_CHECK(cnnlCreatePoolingDescriptor(&pool.pool_desc));
    CNNL_CHECK(cnnlSetPooling2dDescriptor(pool.pool_desc, model_type, maxpooling_nan_opt,
                                          window_height, window_width, pad_t, pad_b, pad_l, pad_r,
                                          vertical_stride, horizon_stride));
    CNNL_CHECK(cnnlGetPoolingWorkspaceSize(ctx->device_ctx()->cambricon_handle(), model_type,
                                           output_dim_w, output_dim_h, &(pool.workspace_size)));
    if (pool.workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc(&(pool.workspace), pool.workspace_size));
      CNRT_CHECK(cnrtMemset(pool.workspace, 0, pool.workspace_size));
    }
    CNNL_CHECK(cnnlPoolingForward(ctx->device_ctx()->cambricon_handle(), pool.pool_desc, nullptr,
                                  pool.input_desc, x_ptr, nullptr, pool.output_desc, y_ptr,
                                  pool.workspace, pool.workspace_size));
    if (pool.workspace != nullptr) { CNRT_CHECK(cnrtFree(pool.workspace)); }
    CNNL_CHECK(cnnlDestroyPoolingDescriptor(pool.pool_desc));
  }
};

#define REGISTER_MAXPOOL_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("max_pool_2d")                       \
      .SetCreateFn<MaxPoolKernelCambricon<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));

#define REGISTER_AVGPOOL_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("avg_pool_2d")                       \
      .SetCreateFn<AvgPoolKernelCambricon<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));

#ifdef WITH_CAMBRICON
REGISTER_MAXPOOL_KERNEL(DeviceType::kCambricon, CamDataType::kFLOAT32)
REGISTER_AVGPOOL_KERNEL(DeviceType::kCambricon, CamDataType::kFLOAT32)
#endif

}  // namespace oneflow
#endif
