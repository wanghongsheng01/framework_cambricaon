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

#include <stdexcept>
#include "oneflow/core/kernel/mlu_tools.h"

namespace oneflow {

size_t dataSize(cnnlDataType_t dtype) {
  switch (dtype) {
    case CNNL_DTYPE_HALF: return 2;
    case CNNL_DTYPE_FLOAT: return 4;
    case CNNL_DTYPE_INT8: return 1;
    case CNNL_DTYPE_INT16: return 2;
    case CNNL_DTYPE_INT32: return 4;
    default: throw std::runtime_error("unsupport data  dtype\n");
  }
}

cnnlDataType_t convert(CamDataType type) {
  int v = 0;
  if (type == kHALF) {
    v = 1;
  } else if (type == kFLOAT32) {
    v = 2;
  } else if (type == kINT8) {
    v = 3;
  } else if (type == kINT16) {
    v = 4;
  }
  return (cnnlDataType_t)v;
}

void set_tensor_desc(cnnlTensorDescriptor_t& desc, size_t ndim, const int* dim,
                     cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, static_cast<int>(ndim), dim));
}

void set_tensor_desc(cnnlTensorDescriptor_t& desc, int dim_n, int dim_h, int dim_w, int dim_c,
                     cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  int dim[4];
  dim[0] = dim_n;
  dim[1] = dim_h;
  dim[2] = dim_w;
  dim[3] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 4, dim));
}

void set_tensor_desc_3d(cnnlTensorDescriptor_t& desc, int dim_0, int dim_1, int dim_2,
                        cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  int dim[3];
  dim[0] = dim_0;
  dim[1] = dim_1;
  dim[2] = dim_2;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 3, dim));
}

void set_tensor_desc_batchnorm(cnnlTensorDescriptor_t& desc, int dim_c, cnnlDataType_t dtype,
                               cnnlTensorLayout_t layout) {
  int dim[1];
  dim[0] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 1, dim));
}

void set_tensor_desc_softmax(cnnlTensorDescriptor_t& desc, int dim_n, int dim_c,
                             cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  int dim[3];
  dim[0] = dim_n;
  dim[1] = 1;
  dim[2] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 3, dim));
}

void set_tensor_desc_matmul(cnnlTensorDescriptor_t& desc, int dim_n, int dim_c,
                            cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  int dim[2];
  dim[0] = dim_n;
  dim[1] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 2, dim));
}

void set_tensor_desc_biasadd(cnnlTensorDescriptor_t& desc, int dim_c, cnnlDataType_t dtype,
                             cnnlTensorLayout_t layout) {
  int dim[1];
  dim[0] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 1, dim));
}

cnrtDataType_t convertCnnlDtypeToCnrt(cnnlDataType_t dtype) {
  switch (dtype) {
    case CNNL_DTYPE_HALF: return CNRT_FLOAT16;
    case CNNL_DTYPE_FLOAT: return CNRT_FLOAT32;
    case CNNL_DTYPE_INT8: return CNRT_INT8;
    case CNNL_DTYPE_INT16: return CNRT_INT16;
    case CNNL_DTYPE_INT32: return CNRT_INT32;
    case CNNL_DTYPE_BOOL: return CNRT_BOOL;
    case CNNL_DTYPE_UINT8: return CNRT_UINT8;
    default: throw std::runtime_error("unsupport data dtype\n");
  }
}

void getPosition(float* input, size_t num, cnnlDataType_t datatype, int* position) {
  if (input == nullptr || position == nullptr || num <= 0) { printf("invalid input paramter!\n"); }

  int bitwidth = 8;
  if (datatype == CNNL_DTYPE_INT8) {
    bitwidth = 8;
  } else if (datatype == CNNL_DTYPE_INT16) {
    bitwidth = 16;
  } else {
    printf("unsuport quant datatype!\n");
  }
  // Formula: int8 int16,  position = ceil(log2(absmax/(2^(bitwidth-1)-1)))
  float absmax = std::fabs(input[0]);
  for (size_t index = 0; index < num; ++index) {
    if (std::fabs(input[index]) > absmax) absmax = std::fabs(input[index]);
  }
  if (absmax == 0) {
    *position = 0;
  } else {
    *position = static_cast<int>(std::floor(std::log2(absmax)) - (bitwidth - 2));
  }
}

void getPositionAndScale(float* input, size_t size, cnnlDataType_t dtype, int* position,
                         float* scale, int mode) {
  if (input == NULL || size == 0 || position == NULL || scale == NULL) {
    printf("invalid input paramter!");
  }

  int bitwidth = 8;
  if (dtype == CNNL_DTYPE_INT8) {
    bitwidth = 8;
  } else if (dtype == CNNL_DTYPE_INT16) {
    bitwidth = 16;
  } else {
    printf("unsupport input data type!");
    return;
  }

  int scale_var = std::pow(2, bitwidth - 1) - 1;
  float max_data = std::fabs(input[0]);
  for (size_t index = 0; index < size; ++index) {
    if (std::fabs(input[index]) > max_data) max_data = std::fabs(input[index]);
  }
  if (max_data == 0) {
    *position = 0;
    *scale = 1.0;
  } else {
    if (mode == 0) {
      *position = static_cast<int>(std::floor(std::log2(max_data)) - (bitwidth - 2));
      *scale = static_cast<float>(std::pow(2, *position) * scale_var / max_data);
    } else {
      *position = static_cast<int>(std::floor(std::log2(max_data)) - (bitwidth - 2));
      *scale = 1.0;
    }
  }
}

void cast_data(float* src_data, cnnlDataType_t src_dtype, char* dst_data, cnnlDataType_t dst_dtype,
               size_t size, int* pos, float* scale, int* offset, int quant_mode) {
  if (src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_FLOAT) {
    memcpy(dst_data, src_data, size * sizeof(float));
  } else if ((src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_INT8)
             || (src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_INT16)) {
    auto in_dtype = convertCnnlDtypeToCnrt(src_dtype);
    auto out_dtype = convertCnnlDtypeToCnrt(dst_dtype);
    // need quant
    *pos = 0;
    *scale = 1.0;
    *offset = 0;

    if (0 == quant_mode) {
      getPosition(src_data, size, dst_dtype, pos);
    } else if (1 == quant_mode) {
      getPositionAndScale(src_data, size, dst_dtype, pos, scale, 1);
    } else {
      printf("This quant mode is not supported at present.");
    }
    cnrtQuantizedParam_t quant_param = nullptr;
    CNRT_CHECK(cnrtCreateQuantizedParam(&quant_param, *pos, *scale, *offset));
    CNRT_CHECK(cnrtCastDataType(src_data, in_dtype, dst_data, out_dtype, size, quant_param));
    CNRT_CHECK(cnrtDestroyQuantizedParam(quant_param));
  } else if ((src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_HALF)
             || (src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_INT32)
             || (src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_BOOL)) {
    auto in_dtype = convertCnnlDtypeToCnrt(src_dtype);
    auto out_dtype = convertCnnlDtypeToCnrt(dst_dtype);
    CNRT_CHECK(cnrtCastDataType(src_data, in_dtype, dst_data, out_dtype, size, nullptr));
  } else {
    printf("This dtype is not supported.\n");
  }
}

}  // namespace oneflow

#endif
