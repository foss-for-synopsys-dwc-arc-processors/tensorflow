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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/arc/scratch_buffers.h"
#include "tensorflow/lite/micro/kernels/arc/scratch_buf_mgr.h"
#include "tensorflow/lite/micro/kernels/arc/mli_tf_utils.h"
#include "tensorflow/lite/micro/kernels/arc/mli_slicers.h"

#include "mli_api.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pooling {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  TfLitePaddingValues padding;
};

TfLiteStatus CalculateOpData(const TfLiteContext* context,
                             const TfLitePoolParams* params,
                             const TfLiteTensor* input,
                             const TfLiteTensor* output, OpData* data) {
  // input: batch, height, width, channel
  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  int out_height, out_width;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1, height, width, params->filter_height,
      params->filter_width, params->padding, &out_height, &out_width);

  return kTfLiteOk;
}

void AverageEvalFloat(const TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;
  reference_ops::AveragePool(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(output), GetTensorData<float>(output));
}

void AverageEvalUint8(TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_ops::AveragePool(
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(output), GetTensorData<uint8_t>(output));
}

TfLiteStatus AverageEvalInt8(TfLiteContext* context, const TfLiteNode* node,
                     const TfLitePoolParams* params, const OpData* data,
                     const TfLiteTensor* input, TfLiteTensor* output) {
  // Run Average Pooling MLI kernel
  // MLI optimized version only supports int8 dataype and no fused Relu
  // TODO: subject to add mli_saturate kernel
  if (input->type == kTfLiteInt8 && params->activation == kTfLiteActNone) {
    mli_tensor mli_in = {0};
    mli_tensor mli_out = {0};
    mli_pool_cfg cfg = {0};

    ConvertToMliTensor<int8_t>(input, &mli_in);
    ConvertToMliTensor<int8_t>(output, &mli_out);

    cfg.kernel_width = params->filter_width;
    cfg.kernel_height = params->filter_height;
    cfg.stride_width = params->stride_width;
    cfg.stride_height = params->stride_height;

    if (params->padding == kTfLitePaddingValid) {
      cfg.padding_left = 0;
      cfg.padding_right = 0;
      cfg.padding_top = 0;
      cfg.padding_bottom = 0;
    } else {
      cfg.padding_left = data->padding.width;
      cfg.padding_right = data->padding.width + data->padding.width_offset;
      cfg.padding_top = data->padding.height;
      cfg.padding_bottom = data->padding.height + data->padding.height_offset;
    }

    mli_point_to_subtsr_cfg subtsr_cfg_in = {{0,0}, 2, static_cast<uint8_t>(mli_in.shape[1])};
    mli_point_to_subtsr_cfg subtsr_cfg_out = {{0,0}, 2, static_cast<uint8_t>(mli_out.shape[1])};
    mli_tensor sub_mli_in = {0};
    mli_tensor sub_mli_out = {0};
    mli_hlp_point_to_subtensor(&mli_in, &subtsr_cfg_in, &sub_mli_in);
    mli_hlp_point_to_subtensor(&mli_out, &subtsr_cfg_out, &sub_mli_out);

    const int height_dimension = 1;
    int in_slice_height = 0;
    int out_slice_height = 0;
    const int overlap = cfg.kernel_height - cfg.stride_height;

    // Tensors for data in fast (local) memory and config to copy data from external to local memory
    mli_tensor in_local = sub_mli_in;
    mli_tensor out_local = sub_mli_out;
    mli_mov_cfg_t copy_config;
    mli_mov_cfg_for_copy(&copy_config);
    TF_LITE_ENSURE_STATUS(get_arc_scratch_buffer_for_pooling_tensors(context, &in_local, &out_local));
    bool in_is_local = in_local.data == sub_mli_in.data;
    bool out_is_local = out_local.data == sub_mli_out.data;
    TF_LITE_ENSURE_STATUS(arc_scratch_buffer_calc_slice_size_io(&in_local, &out_local, cfg.kernel_height, cfg.stride_height, cfg.padding_top, cfg.padding_bottom, &in_slice_height, &out_slice_height));

    /* mli_in tensor contains batches of HWC tensors. so it is a 4 dimensional tensor.
       because the mli kernel will process one HWC tensor at a time, the 4 dimensional tensor needs to be sliced into nBatch 3 dimensional tensors.
       on top of that there could be a need to also slice in the Height dimension. for that the sliceHeight has been calculated.
       The tensor slicer is configured that it will completely slice the nBatch dimension (0) and slice the height dimension (1)
       in chunks of 'sliceHeight' */
    TensorSlicer in_slice(&mli_in, height_dimension, in_slice_height, cfg.padding_top, cfg.padding_bottom, overlap);
    TensorSlicer out_slice(&mli_out, height_dimension, out_slice_height);

    /* is_local indicates that the tensor is already in local memory,
       so in that case the original tensor can be used,
       and there is no need to copy it to the local tensor*/
    mli_tensor *in_ptr = in_is_local ? in_slice.Sub() : &in_local;
    mli_tensor *out_ptr = out_is_local ? out_slice.Sub() : &out_local;

    while (!out_slice.Done()) {
      cfg.padding_top = in_slice.GetPaddingPre();
      cfg.padding_bottom = in_slice.GetPaddingPost();

      mli_mov_tensor_sync(in_slice.Sub(), &copy_config, in_ptr);
      mli_krn_avepool_hwc_sa8(in_ptr, &cfg, out_ptr);
      mli_mov_tensor_sync(out_ptr, &copy_config, out_slice.Sub());

      in_slice.Next();
      out_slice.Next();
    }

  } else {
    int32_t activation_min, activation_max;
    (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                            &activation_min, &activation_max);
    PoolParams op_params;
    op_params.stride_height = params->stride_height;
    op_params.stride_width = params->stride_width;
    op_params.filter_height = params->filter_height;
    op_params.filter_width = params->filter_width;
    op_params.padding_values.height = data->padding.height;
    op_params.padding_values.width = data->padding.width;
    op_params.quantized_activation_min = activation_min;
    op_params.quantized_activation_max = activation_max;
    reference_integer_ops::AveragePool(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  }
  return kTfLiteOk;
}

void MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLitePoolParams* params, OpData* data,
                  const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;
  reference_ops::MaxPool(op_params, GetTensorShape(input),
                         GetTensorData<float>(input), GetTensorShape(output),
                         GetTensorData<float>(output));
}

void MaxEvalQuantizedUInt8(TfLiteContext* context, TfLiteNode* node,
                           TfLitePoolParams* params, OpData* data,
                           const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_ops::MaxPool(op_params, GetTensorShape(input),
                         GetTensorData<uint8_t>(input), GetTensorShape(output),
                         GetTensorData<uint8_t>(output));
}

}  // namespace

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData data;

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, &data));

  // Inputs and outputs share the same type, guarenteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AverageEvalFloat(context, node, params, &data, input, output);
      break;
    case kTfLiteUInt8:
      AverageEvalUint8(context, node, params, &data, input, output);
      break;
    case kTfLiteInt8:
      return AverageEvalInt8(context, node, params, &data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input type %s is not currently supported",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData data;

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, &data));

  switch (input->type) {
    case kTfLiteFloat32:
      MaxEvalFloat(context, node, params, &data, input, output);
      break;
    case kTfLiteUInt8:
      MaxEvalQuantizedUInt8(context, node, params, &data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration* Register_AVERAGE_POOL_2D() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/pooling::AverageEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_2D() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/pooling::MaxEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
