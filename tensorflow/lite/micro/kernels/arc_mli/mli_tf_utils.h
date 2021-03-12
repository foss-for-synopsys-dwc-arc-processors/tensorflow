/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

constexpr int kFracBitsQ15 = 15;
constexpr int kFracBitsQ31 = 31;

namespace tflite {
namespace ops {
namespace micro {

inline void ConvertToMliTensorData(const TfLiteTensor* tfT, mli_tensor* mliT) {
  // Data is NULL until MliTensorAttachBuffer is called.
  mliT->data.mem.void_p = nullptr;
  if (tfT->type == kTfLiteInt8) {
    mliT->el_type = MLI_EL_SA_8;
  } else if (tfT->type == kTfLiteInt32) {
    mliT->el_type = MLI_EL_SA_32;
  } else {
    TF_LITE_FATAL("Wrong data type. Expected int8_t or int32_t.");
  }

  mliT->data.capacity = tfT->bytes;
  mliT->rank = GetTensorShape(tfT).DimensionsCount();
  for (int i = 0; i < GetTensorShape(tfT).DimensionsCount(); i++) {
    mliT->shape[i] = GetTensorShape(tfT).Dims(i);
  }
}

inline void ConvertToMliQuantParams(const TfLiteTensor* tfT, mli_tensor* mliT) {
  mliT->el_params.sa.dim = -1;
  mliT->el_params.sa.zero_point.mem.i16 = tfT->params.zero_point;
  float fscale = tfT->params.scale;
  int exp;
  frexpf(fscale, &exp);
  int frac_bits = kFracBitsQ15 - exp;
  int16_t iscale = (int16_t)((1ll << frac_bits) * fscale + 0.5f);
  mliT->el_params.sa.scale_frac_bits.mem.i8 = frac_bits;
  mliT->el_params.sa.scale.mem.i16 = (int16_t)iscale;
}

inline void ConvertToMliQuantParamsPerChannel(const TfLiteTensor* tfT,
                                              mli_tensor* mliT,
                                              bool is_bias_tensor) {
  // mli tensor scale and zero_point arrays should be allocated at this point
  TFLITE_DCHECK_NE(mliT->el_params.sa.scale.mem.pi16, 0);
  TFLITE_DCHECK_NE(mliT->el_params.sa.zero_point.mem.pi16, 0);

  // get per channel quantization parameters
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(tfT->quantization.params);
  mliT->el_params.sa.dim =
      is_bias_tensor ? 0 : affine_quantization->quantized_dimension;

  // find frac_bits
  const int num_channels =
      is_bias_tensor ? mliT->shape[0]
                     : mliT->shape[affine_quantization->quantized_dimension];
  // int min_frac_bits;
  float* fscale = affine_quantization->scale->data;
  for (int i = 0; i < num_channels; i++) {
    int exp;
    frexpf(fscale[i], &exp);
    int cur_frac_bits = kFracBitsQ15 - exp;
    mliT->el_params.sa.scale_frac_bits.mem.pi8[i] = cur_frac_bits;
  }

  for (int i = 0; i < num_channels; i++) {
    int16_t iscale = (int16_t)(
        (1ll << mliT->el_params.sa.scale_frac_bits.mem.pi8[i]) * fscale[i] +
        0.5f);
    mliT->el_params.sa.scale.mem.pi16[i] = iscale;
    mliT->el_params.sa.zero_point.mem.pi16[i] = tfT->params.zero_point;
  }
}

template <typename datatype>
inline void MliTensorAttachBuffer(const TfLiteEvalTensor* tfT,
                                  mli_tensor* mliT) {
  // "const_cast" here used to attach const data buffer to the initially
  // non-const mli_tensor. This is required by current implementation of MLI
  // backend and planned for redesign due to this and some other aspects.
  mliT->data.mem.void_p = const_cast<void*>(
      static_cast<const void*>(tflite::micro::GetTensorData<datatype>(tfT)));
}

inline void ConvertToMliTensor(const TfLiteTensor* tfT, mli_tensor* mliT) {
  ConvertToMliTensorData(tfT, mliT);
  ConvertToMliQuantParams(tfT, mliT);
}

inline void ConvertToMliTensorPerChannel(const TfLiteTensor* tfT,
                                         mli_tensor* mliT,
                                         bool is_bias_tensor) {
  ConvertToMliTensorData(tfT, mliT);
  ConvertToMliQuantParamsPerChannel(tfT, mliT, is_bias_tensor);
}

template <typename datatype>
inline void ConvertCHWNToHWCN(mli_tensor* mliT) {
  // Extend shape to the MLI_MAX_RANK complimenting it with 1s in a front.
  // Calculate strides on input float array and output tensor
  // for easier definition of element position in total arrays
  uint32_t mli_tensor_strides_prev[MLI_MAX_RANK] = {0};
  int shape_idx = mliT->rank - 1;
  int mli_tensor_memstride = 1;
  for (; shape_idx >= 0; --shape_idx) {
    mli_tensor_strides_prev[shape_idx] = (mliT->mem_stride[shape_idx] == 0)
                                             ? mli_tensor_memstride
                                             : mliT->mem_stride[shape_idx];
    mli_tensor_memstride *= mliT->shape[shape_idx];
  }

  auto reorder = [](uint32_t* arr, int index[]) {
    uint32_t temp[MLI_MAX_RANK];
    for (int i = 0; i < MLI_MAX_RANK; i++) temp[index[i]] = arr[i];
    for (int i = 0; i < MLI_MAX_RANK; i++) {
      arr[i] = temp[i];
      index[i] = i;
    }
  };

  // Change shape to HWCN
  reorder(mliT->shape, (int[]){2, 0, 1, 3});

  // Calculate strides for new layout
  shape_idx = mliT->rank - 1;
  mli_tensor_memstride = 1;
  for (; shape_idx >= 0; --shape_idx) {
    mliT->mem_stride[shape_idx] = mli_tensor_memstride;
    mli_tensor_memstride *= mliT->shape[shape_idx];
  }
}

template <typename datatype>
inline void ConvertNHWCToHWCN(mli_tensor* mliT, void* mliT_storage) {
  datatype* data_ptr = (datatype*)mliT->data.mem.void_p;
  datatype* storage_data_ptr = (datatype*)mliT_storage;

  // Extend shape to the MLI_MAX_RANK complimenting it with 1s in a front.
  // Calculate strides on input float array and output tensor
  // for easier definition of element position in total arrays
  uint32_t mli_tensor_strides_prev[MLI_MAX_RANK] = {0};
  int shape_idx = mliT->rank - 1;
  int mli_tensor_memstride = 1;
  for (; shape_idx >= 0; --shape_idx) {
    mli_tensor_strides_prev[shape_idx] = (mliT->mem_stride[shape_idx] == 0)
                                             ? mli_tensor_memstride
                                             : mliT->mem_stride[shape_idx];
    mli_tensor_memstride *= mliT->shape[shape_idx];
  }

  auto val_pos = [](uint32_t strides[MLI_MAX_RANK], int dim0_idx, int dim1_idx,
                    int dim2_idx, int dim3_idx) -> int {
    return (strides[0] * dim0_idx) + (strides[1] * dim1_idx) +
           (strides[2] * dim2_idx) + (strides[3] * dim3_idx);
  };

  auto reorder = [](uint32_t* arr, int index[]) {
    uint32_t temp[MLI_MAX_RANK];
    for (int i = 0; i < MLI_MAX_RANK; i++) temp[index[i]] = arr[i];
    for (int i = 0; i < MLI_MAX_RANK; i++) {
      arr[i] = temp[i];
      index[i] = i;
    }
  };

  int dim_start[MLI_MAX_RANK] = {0};
  int dim_end[MLI_MAX_RANK] = {0};
  for (int i = 0; i < MLI_MAX_RANK; ++i) {
    dim_end[i] = mliT->shape[i];
  }

  // Change shape to HWCN
  reorder(mliT->shape, (int[]){3, 0, 1, 2});

  if (mliT->el_params.sa.dim > -1) {
    // TODO: change this hardcode
    mliT->el_params.sa.dim = 3;
  }

  // Calculate strides for new layout
  shape_idx = mliT->rank - 1;
  mli_tensor_memstride = 1;
  for (; shape_idx >= 0; --shape_idx) {
    mliT->mem_stride[shape_idx] = mli_tensor_memstride;
    mli_tensor_memstride *= mliT->shape[shape_idx];
  }

  // Apply transformation of defined slice
  for (int dim0_idx = dim_start[0]; dim0_idx < dim_end[0]; ++dim0_idx) {
    for (int dim1_idx = dim_start[1]; dim1_idx < dim_end[1]; ++dim1_idx) {
      for (int dim2_idx = dim_start[2]; dim2_idx < dim_end[2]; ++dim2_idx) {
        for (int dim3_idx = dim_start[3]; dim3_idx < dim_end[3]; ++dim3_idx) {
          uint16_t prev_position = val_pos(mli_tensor_strides_prev, dim0_idx,
                                           dim1_idx, dim2_idx, dim3_idx);
          uint16_t new_position =
              val_pos(mliT->mem_stride, dim1_idx, dim2_idx, dim3_idx, dim0_idx);
          storage_data_ptr[new_position] = data_ptr[prev_position];
        }
      }
    }
  }

  mliT->data.mem.void_p = storage_data_ptr;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_
