# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# Settings for EMSDP target (ARC processor)

TARGET_ARCH := arc
ARC_TOOLCHAIN ?= mwdt


BUILD_ARC_MLI := false
ARC_MLI_PRE_COMPILED_TARGET := emsdp_em11d_em9d_dfss

ifneq ($(filter no_arc_mli,$(ALL_TAGS)),)
  MLI_LIB_DIR = arc_mli_package
  MLI_GNU_DEPEND_LIB_DIR = arc_gnu_dependencies
  $(eval $(call add_third_party_download,$(EMBARC_MLI_PRE_COMPILED_URL),$(EMBARC_MLI_PRE_COMPILED_MD5),$(MLI_LIB_DIR),))
  $(eval $(call add_third_party_download,$(EMBARC_GNU_DEPENDS_PRE_COMPILED_URL),$(EMBARC_GNU_DEPENDS_PRE_COMPILED_MD5),$(MLI_GNU_DEPEND_LIB_DIR),))
else ifeq ($(BUILD_ARC_MLI), true)
  MLI_LIB_DIR = arc_mli_$(ARC_MLI_PRE_COMPILED_TARGET)
endif

ifeq ($(ARC_TOOLCHAIN), mwdt)
  TCF_FILE = $(PWD)/$(MAKEFILE_DIR)/downloads/$(MLI_LIB_DIR)/hw/emsdp_em11d_em9d_dfss.tcf
  LCF_FILE = $(PWD)/$(MAKEFILE_DIR)/targets/arc/emsdp/emsdp.lcf

else ifeq ($(ARC_TOOLCHAIN), gnu)
  LCF_FILE = $(PWD)/$(MAKEFILE_DIR)/targets/arc/emsdp/memory.x
  ARC_HEADERS += $(PWD)/$(MAKEFILE_DIR)/targets/arc/emsdp/gnu_dependences/core_config.h
endif # ARC_TOOLCHAIN

include $(MAKEFILE_DIR)/targets/arc/arc_common.inc
  
  ARC_EXTRA_APP_SETTINGS = \
    BIN_DIR = .$(DLR)\(PS\)bin\n\
    BIN_FILE = $(DLR)\(BIN_DIR\)$(DLR)\(PS\)app.elf\n

  ARC_EXTRA_APP_RULES = \
    $(DLR)\(BIN_FILE\): $(DLR)\(BIN_DIR\) $(DLR)\(OUT_NAME\)\
    \n\t\@$(DLR)\(CP\) $(DLR)\(OUT_NAME\) $(DLR)\(BIN_FILE\)\
    \n \
    \n$(DLR)\(BIN_DIR\):\
    \n\t\@$(DLR)\(MKDIR\) $(DLR)\(BIN_DIR\)\

  ARC_EXTRA_RM_TARGETS = $(DLR)\(BIN_DIR\)

  ARC_BIN_DEPEND = $(DLR)\(BIN_DIR\) $(DLR)\(BIN_FILE\)
  ARC_BIN_RULE = \t@echo Copy content of $(DLR)\(BIN_DIR\) into the root of SD card and follow instructions

ifeq ($(ARC_TOOLCHAIN), mwdt)
  ARC_APP_RUN_CMD = mdb -run -digilent -nooptions $(DLR)\(DBG_ARGS\)
  ARC_APP_DEBUG_CMD = mdb -OK -digilent -nooptions $(DLR)\(DBG_ARGS\)
  ARC_EXTRA_EXECUTE_RULES = 
else ifeq ($(ARC_TOOLCHAIN), gnu)
  ARC_APP_RUN_CMD = mdb -run -digilent -nooptions -off=spot_verify_download $(DLR)\(DBG_ARGS\)
  ARC_APP_DEBUG_CMD = mdb -OK -digilent -nooptions -off=spot_verify_download $(DLR)\(DBG_ARGS\)
  ARC_EXTRA_EXECUTE_RULES = 
endif

MAKE_PROJECT_FILES := $(filter-out README_MAKE.md, $(MAKE_PROJECT_FILES)) README_ARC_EMSDP.md

# for default EMSDP configuration we can use em9d_va rt libs
# for better performance runtime should be built for emsdp configuration
# No hostlink library for smaller codesize purpose

ifeq ($(ARC_TOOLCHAIN), mwdt)
  PLATFORM_LDFLAGS += -Hlib=em9d_voice_audio -Hhostlib=
  
else ifeq ($(ARC_TOOLCHAIN), gnu)
  CCFLAGS += -mcpu=em4_fpuda -mlittle-endian -mcode-density -mdiv-rem -mswap -mnorm -mmpy-option=6 -mbarrel-shifter -mfpu=fpuda_all --param l1-cache-size=16384 --param l1-cache-line-size=32
  CXXFLAGS += -mcpu=em4_fpuda -mlittle-endian -mcode-density -mdiv-rem -mswap -mnorm -mmpy-option=6 -mbarrel-shifter -mfpu=fpuda_all --param l1-cache-size=16384 --param l1-cache-line-size=32

endif # ARC_TOOLCHAIN
