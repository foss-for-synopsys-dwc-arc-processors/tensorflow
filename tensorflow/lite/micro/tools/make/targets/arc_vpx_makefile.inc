# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Settings for VPX target (ARC VPX processor)

TARGET_ARCH := arc
ARC_TOOLCHAIN := mwdt

BUILD_ARC_MLI := true
ARC_MLI_TARGET := vpx5_integer_full

MLI_LIB_DIR = arc_mli_$(ARC_MLI_TARGET)

TCF_FILE = $(PWD)/$(MAKEFILE_DIR)/downloads/$(MLI_LIB_DIR)/hw/vpx5_integer_full.tcf
BUILD_LIB_DIR := vpx5_integer_full


include $(MAKEFILE_DIR)/targets/arc/arc_common.inc

LDFLAGS += -Hlib=vpx5_integer_full
CXXFLAGS += -DMLI_2_0
CCFLAGS += -DMLI_2_0

MAKE_PROJECT_FILES := $(filter-out README_MAKE.md, $(MAKE_PROJECT_FILES)) README_ARC.md

