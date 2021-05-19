/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mli_interface.h"  // NOLINT

#include <math.h>

namespace tflite {
namespace ops {
namespace micro {

#ifdef MLI_2_0
template <>
int8_t** MliTensorInterface::Data(void) {
  return &tensor_->data.mem.pi8;
}

template <>
int32_t** MliTensorInterface::Data(void) {
  return &tensor_->data.mem.pi32;
}

template <>
int16_t** MliTensorInterface::Scale(void) {
  return &tensor_->el_params.sa.scale.mem.pi16;
}

template <>
int16_t* MliTensorInterface::Scale(void) {
  return &tensor_->el_params.sa.scale.mem.i16;
}

#else

template <>
void** MliTensorInterface::Data<int8_t>(void) {
  return &tensor_->data;
}

template <>
void** MliTensorInterface::Data<int32_t>(void) {
  return &tensor_->data;
}

template <>
int32_t* MliTensorInterface::Scale(void) {
  return &tensor_->el_params.asym.scale.i32;
}

template <>
int32_t** MliTensorInterface::Scale(void) {
  return &tensor_->el_params.asym.scale.pi32;
}
#endif

template <>
void MliTensorInterface::SetData(int8_t* data) const {
#ifdef MLI_2_0
  tensor_->data.mem.pi8 = data;
#else
  tensor_->data = data;
#endif
}

template <>
void MliTensorInterface::SetData(int32_t* data) const {
#ifdef MLI_2_0
  tensor_->data.mem.pi32 = data;
#else
  tensor_->data = data;
#endif
}

mli_tensor* MliTensorInterface::MliTensor(void) { return tensor_; }

const mli_tensor* MliTensorInterface::MliTensor(void) const {
  return static_cast<const mli_tensor*>(
      const_cast<MliTensorInterface*>(this)->MliTensor());
}

uint32_t* MliTensorInterface::Rank(void) { return &tensor_->rank; }

uint32_t* MliTensorInterface::DataCapacity(void) {
#ifdef MLI_2_0
  return &tensor_->data.capacity;
#else
  return &tensor_->capacity;
#endif
}

const uint32_t* MliTensorInterface::DataCapacity(void) const {
  return static_cast<const uint32_t*>(
      const_cast<MliTensorInterface*>(this)->DataCapacity());
}

mli_element_type* MliTensorInterface::ElType(void) { return &tensor_->el_type; }

template <>
int16_t* MliTensorInterface::ZeroPoint(void) {
#ifdef MLI_2_0
  return &tensor_->el_params.sa.zero_point.mem.i16;
#else
  return &tensor_->el_params.asym.zero_point.i16;
#endif
}

template <>
int16_t** MliTensorInterface::ZeroPoint(void) {
#ifdef MLI_2_0
  return &tensor_->el_params.sa.zero_point.mem.pi16;
#else
  return &tensor_->el_params.asym.zero_point.pi16;
#endif
}

uint32_t* MliTensorInterface::ZeroPointCapacity(void) {
#ifdef MLI_2_0
  return &tensor_->el_params.sa.zero_point.capacity;
#else
  return nullptr;
#endif
}

int32_t* MliTensorInterface::Dim(void) {
#ifdef MLI_2_0
  return &tensor_->el_params.sa.dim;
#else
  return &tensor_->el_params.asym.dim;
#endif
}

uint32_t* MliTensorInterface::ScaleCapacity(void) {
#ifdef MLI_2_0
  return &tensor_->el_params.sa.scale.capacity;
#else
  return nullptr;
#endif
}

template <>
int8_t** MliTensorInterface::ScaleFracBits(void) {
#ifdef MLI_2_0
  return &tensor_->el_params.sa.scale_frac_bits.mem.pi8;
#else
  return nullptr;
#endif
}

template <>
int8_t* MliTensorInterface::ScaleFracBits(void) {
#ifdef MLI_2_0
  return &tensor_->el_params.sa.scale_frac_bits.mem.i8;
#else
  return &tensor_->el_params.asym.scale_frac_bits;
#endif
}

uint32_t* MliTensorInterface::ScaleFracBitsCapacity(void) {
#ifdef MLI_2_0
  return &tensor_->el_params.sa.scale_frac_bits.capacity;
#else
  return nullptr;
#endif
}

int32_t* MliTensorInterface::MemStride(void) { return tensor_->mem_stride; }

uint32_t* MliTensorInterface::Shape(void) { return tensor_->shape; }

const uint32_t* MliTensorInterface::Shape(void) const {
  return static_cast<const uint32_t*>(
      const_cast<MliTensorInterface*>(this)->Shape());
}

void MliTensorInterface::SetScale(float fscale) {
  int exp;
  frexpf(fscale, &exp);
#ifdef MLI_2_0
  int frac_bits = 15 - exp;
  int16_t iscale = (int16_t)((1ll << frac_bits) * fscale + 0.5f);
  *(this->Scale<int16_t*>()) = (int16_t)iscale;
  *(this->ScaleFracBits<int8_t*>()) = frac_bits;
  *this->ScaleCapacity() = 1 * sizeof(int16_t);
  *this->ScaleFracBitsCapacity() = 1 * sizeof(int8_t);
#else
  int frac_bits = 31 - exp;
  int32_t iscale = (int32_t)((1ll << frac_bits) * fscale + 0.5f);
  *(this->ScaleFracBits<int8_t*>()) = frac_bits;
  *(this->Scale<int32_t*>()) = (int32_t)iscale;
#endif
}

void MliTensorInterface::SetScalePerChannel(float* fscale,
                                            const int num_channels) {
#ifdef MLI_2_0
  for (int i = 0; i < num_channels; i++) {
    int exp;
    frexpf(fscale[i], &exp);
    int cur_frac_bits = 15 - exp;
    (*this->ScaleFracBits<int8_t**>())[i] = cur_frac_bits;
  }

  for (int i = 0; i < num_channels; i++) {
    int16_t iscale = (int16_t)(
        (1ll << (*this->ScaleFracBits<int8_t**>())[i]) * fscale[i] + 0.5f);
    (*this->Scale<int16_t**>())[i] = iscale;
  }
#else
  int min_frac_bits;
  for (int i = 0; i < num_channels; i++) {
    int exp;
    frexpf(fscale[i], &exp);
    int cur_frac_bits = 31 - exp;
    if (i == 0) {
      min_frac_bits = cur_frac_bits;
    } else {
      min_frac_bits =
          min_frac_bits < cur_frac_bits ? min_frac_bits : cur_frac_bits;
    }
  }
  *this->ScaleFracBits<int8_t*>() = min_frac_bits;

  for (int i = 0; i < num_channels; i++) {
    int32_t iscale = (int32_t)((1ll << min_frac_bits) * fscale[i] + 0.5f);
    (*this->Scale<int32_t**>())[i] = iscale;
  }

#endif
}

void MliTensorInterface::SetElType(TfLiteType type) {
#ifdef MLI_2_0
  if (type == kTfLiteInt8) {
    *this->Data<int8_t>() = nullptr;
    *this->ElType() = MLI_EL_SA_8;
  } else if (type == kTfLiteInt32) {
    *this->Data<int32_t>() = nullptr;
    *this->ElType() = MLI_EL_SA_32;
  }
#else
  if (type == kTfLiteInt8) {
    *this->Data<int8_t>() = nullptr;
    *this->ElType() = MLI_EL_ASYM_I8;
  } else if (type == kTfLiteInt32) {
    *this->Data<int32_t>() = nullptr;
    *this->ElType() = MLI_EL_ASYM_I32;
  }
#endif
  else {
    TF_LITE_FATAL("Wrong data type. Expected int8_t or int32_t.");
  }
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite