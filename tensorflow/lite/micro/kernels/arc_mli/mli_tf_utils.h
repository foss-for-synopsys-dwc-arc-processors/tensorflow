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

inline void change_mem_stride(mli_tensor* mliT, int8_t dim_order[]) {
  auto reorder = [](uint32_t* arr, int8_t index[]) {
    uint32_t temp[MLI_MAX_RANK];
    for (int8_t i = 0; i < MLI_MAX_RANK; i++) temp[index[i]] = arr[i];
    for (int8_t i = 0; i < MLI_MAX_RANK; i++) {
      arr[i] = temp[i];
      index[i] = i;
    }
  };

  reorder(mliT->shape, dim_order);

  if (mliT->el_params.sa.dim > -1 && mliT->rank == MLI_MAX_RANK) {
    mliT->el_params.sa.dim = MLI_MAX_RANK - 1;
  }

  // Calculate strides for new layout
  int mli_tensor_memstride = 1;
  for (int shape_idx = mliT->rank - 1; shape_idx >= 0; --shape_idx) {
    mliT->mem_stride[shape_idx] = mli_tensor_memstride;
    mli_tensor_memstride *= mliT->shape[shape_idx];
  }
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_
