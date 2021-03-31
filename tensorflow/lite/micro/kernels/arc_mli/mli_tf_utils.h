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
  int32_t quantized_dimension =
      is_bias_tensor ? 0 : affine_quantization->quantized_dimension;

  mliT->el_params.sa.dim = quantized_dimension;

  // find frac_bits
  const int num_channels = mliT->shape[quantized_dimension];
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

inline void permute_conv_weights_1x1(const mli_tensor* weights_src,
                            const mli_permute_cfg* permute_cfg,
                            mli_tensor* weights_dst, mli_tensor buffer) {
  buffer.el_params = weights_dst->el_params;
  if (buffer.shape[0] * buffer.shape[1] * buffer.shape[2] >=
      weights_src->shape[1] * weights_src->shape[2] * weights_src->shape[3]) {
    mli_mov_cfg_t copy_config;
    mli_mov_cfg_for_copy(&copy_config);
    mli_mov_tensor_sync(weights_src, &copy_config, &buffer);
    for (int i = 0; i < MLI_MAX_RANK; i++) weights_dst->mem_stride[i] = 0;
    mli_krn_permute_sa8(&buffer, permute_cfg, weights_dst);
  } else {
    uint32_t slice_size = buffer.shape[0] * buffer.shape[1] * buffer.shape[2];
    mli_mov_cfg_t copy_config;
    uint32_t offsets[] = {0, 0, 0, 0};  // TODO
    uint32_t sizes[] = {0, 0, 0, 0};    // TODO
    int dst_mem_stride[] = {0, 0, 0, 0};

    // auto calculate_sizes = [](const mli_tensor* mliT, mli_sub_tensor_cfg* cfg, uint32_t* sizes, const uint32_t slice_size) {
    //   // int mli_tensor_memstride = 1;
    //   // int mli_tensor_memstrides[MLI_MAX_RANK] = { 0 };

    //   // for (int shape_idx = mliT->rank - 1; shape_idx > 0; --shape_idx) {
    //   //   mli_tensor_memstrides[shape_idx] = (mliT->mem_stride[shape_idx] == 0) ? mli_tensor_memstride : mliT->mem_stride[shape_idx];
    //   //   mli_tensor_memstride *= mliT->shape[shape_idx];
    //   // }

    //   uint32_t slice_size_left = slice_size;
    //   for (int i = mliT->rank - 2; i >= 0; --i) {
    //     cfg->size[i] = slice_size_left / mliT->shape[i] > 0 ? mliT->shape[i] : slice_size_left;
    //     slice_size_left /= mliT->shape[i];
    //     slice_size_left = slice_size_left > 0 ? slice_size_left : 1;
    //   }
    //   // cfg->size[0] = slice_size;
    //   // uint32_t temp[MLI_MAX_RANK];
    //   // for (int8_t i = 0; i < MLI_MAX_RANK; i++) temp[index[i]] = arr[i];
    //   // for (int8_t i = 0; i < MLI_MAX_RANK; i++) {
    //   //   arr[i] = temp[i];
    //   //   index[i] = i;
    //   // }
    // };

    // TODO: Change names
    // mli_mov_cfg_t copy_config_2;
    // uint32_t offsets_2[] = {0, 0, 0, 0};  // TODO
    // int sizes_2[] = {0, 0, 0, 0};         // TODO
    // int dst_mem_stride_2[] = {0, 0, 0, 0};
 
    mli_tensor weights_dst_sub_tensor;
    mli_sub_tensor_cfg sub_tensor_cfg = {};
    sub_tensor_cfg.sub_tensor_rank = 4;

    int8_t dim_order[] = {3, 0, 1, 2};
    change_mem_stride(weights_dst, dim_order);

    // TODO: Replace with function which will count this values
    sub_tensor_cfg.size[3] = sizes[0] = buffer.shape[buffer.rank - 1];
    uint32_t slice_size_left = slice_size;
    for (int i = weights_dst->rank - 2; i >= 0; --i) {
      sub_tensor_cfg.size[i] = sizes[i + 1] = slice_size_left / weights_dst->shape[i] > 0 ? weights_dst->shape[i] : slice_size_left;
      slice_size_left /= weights_dst->shape[i];
      slice_size_left = slice_size_left > 0 ? slice_size_left : 1;
    }
    // calculate_sizes(weights_dst, &sub_tensor_cfg, sizes, slice_size);
    // sub_tensor_cfg.size[0] = sizes[1] = 1;
    // sub_tensor_cfg.size[1] = sizes[2] = 1;
    // sub_tensor_cfg.size[2] = sizes[3] = slice_size;

    for (sub_tensor_cfg.offset[3] = offsets[0] = 0; offsets[0] < weights_src->shape[0]; sub_tensor_cfg.offset[3] = offsets[0] += sizes[0], sizes[0] = std::min(sizes[0], weights_src->shape[0] - offsets[0])) {
      for (sub_tensor_cfg.offset[0] = offsets[1] = 0; offsets[1] < weights_src->shape[1]; sub_tensor_cfg.offset[0] = offsets[1] += sizes[1], sizes[1] = std::min(sizes[1], weights_src->shape[1] - offsets[1])) {
        for (sub_tensor_cfg.offset[1] = offsets[2] = 0; offsets[2] < weights_src->shape[2]; sub_tensor_cfg.offset[1] = offsets[2] += sizes[2], sizes[2] = std::min(sizes[2], weights_src->shape[2] - offsets[2])) {
          for (sub_tensor_cfg.offset[2] = offsets[3] = 0; offsets[3] < weights_src->shape[3]; sub_tensor_cfg.offset[2] = offsets[3] += sizes[3], sizes[3] = std::min(sizes[3], weights_src->shape[3] - offsets[3])) {
            mli_mov_cfg_for_slice(&copy_config, (int*)offsets, (int*)sizes, dst_mem_stride);
            mli_mov_tensor_sync(weights_src, &copy_config, &buffer);

            // mli_mov_cfg_for_slice(&copy_config_2, (int*)offsets_2, sizes_2, dst_mem_stride_2);
            // mli_status mli_hlp_create_subtensor(const mli_tensor *in, const mli_sub_tensor_cfg *cfg, mli_tensor *out);
            // typedef struct {
            //     uint32_t offset[MLI_MAX_RANK];   /**< subtensor start coordinates in the input tensor 
            //                                           The size of this array is determined by the rank of the input tensor */
            //     uint32_t size[MLI_MAX_RANK];     /**< Size of the sub tensor in elements per dimension
            //                                           the number of entries in this array is determind by the input tensor */
            //     uint32_t sub_tensor_rank;        /**< Rank of the sub tensor that will be produced */
            // } mli_sub_tensor_cfg;

            mli_hlp_create_subtensor(weights_dst, &sub_tensor_cfg, &weights_dst_sub_tensor);
            //TODO: Do here subtensor and mov there
            mli_krn_permute_sa8(&buffer, permute_cfg, &weights_dst_sub_tensor);
          }
        }
      }
    }

    // for (int i = 1; i <= buffer->shape[3] / slice_size; i++) {
      // sizes[0] = buffer->shape[3];
      // sizes[3] = slice_size;
      // mli_mov_cfg_for_slice(&copy_config, offsets, sizes, dst_mem_stride);
      // mli_mov_tensor_sync(weights_src, &copy_config, buffer);
      // offsets[3] += sizes[3];
    // }

    //(mli_mov_cfg_t* cfg, int* offsets, int* sizes, int* dst_mem_stride);
    // • cfg: pointer to the config structure that will be filled
    // • offsets: Start coordinate in the source tensor. Values must be smaller
    // than the shape of the source tensor.
    // • sizes: Size of the copy in
    // elements per dimension.
    //  • dst_mem_stride: Distance in elements to the
    // next dimension in the destination tensor.
  }

  // mli_sub_tensor_cfg sub_tensor_cfg = {};
  // mli_hlp_create_subtensor(weights_src, &sub_tensor_cfg, buffer);
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_
