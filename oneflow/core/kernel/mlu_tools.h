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
#ifndef ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_
#define ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_

#ifdef WITH_CAMBRICON

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

#include "cnrt.h"
#include "cnnl.h"

namespace oneflow {

enum CamDataType { kHALF, kFLOAT32, kINT32, kINT16, kINT8 };

struct convDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t weight_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct poolDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct reluDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct sigmoidDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct preluDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlDataType_t alpha_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct BatchNormType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlDataType_t weight_bias_mean_var_desc_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct InstanceNormType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlDataType_t scale_bias_desc_dtype;
  cnnlDataType_t mean_var_desc_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct SoftmaxType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct MatMulType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct TransposeType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct InterpType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct AddType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct ConcatType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct BiasAddType {
  cnnlDataType_t a_dtype;
  cnnlDataType_t b_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

size_t dataSize(cnnlDataType_t dtype);

void set_tensor_desc(cnnlTensorDescriptor_t& desc, size_t ndim, const int* dim,
                     cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc(cnnlTensorDescriptor_t& desc, int dim_n, int dim_h, int dim_w, int dim_c,
                     cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc_3d(cnnlTensorDescriptor_t& desc, int dim_0, int dim_1, int dim_2,
                        cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc_batchnorm(cnnlTensorDescriptor_t& desc, int dim_c, cnnlDataType_t dtype,
                               cnnlTensorLayout_t layout);

void set_tensor_desc_softmax(cnnlTensorDescriptor_t& desc, int dim_n, int dim_c,
                             cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc_matmul(cnnlTensorDescriptor_t& desc, int dim_n, int dim_c,
                            cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc_biasadd(cnnlTensorDescriptor_t& desc, int dim_c, cnnlDataType_t dtype,
                             cnnlTensorLayout_t layout);

cnrtDataType_t convertCnnlDtypeToCnrt(cnnlDataType_t dtype);

void getPosition(float* input, size_t num, cnnlDataType_t datatype, int* position);

void getPositionAndScale(float* input, size_t size, cnnlDataType_t dtype, int* pos, float* scale);

void cast_data(float* src_data, cnnlDataType_t src_dtype, char* dst_data, cnnlDataType_t dst_dtype,
               size_t size, int* pos, float* scale, int* offset, int quant_mode);

cnnlDataType_t convert(CamDataType type);

}  // namespace oneflow

#endif

#endif  // ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_
