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

# Settings for IoT Development Kit target (ARC processor)
ifeq ($(TARGET), arc_iotdk)

  TARGET_ARCH := arc
  ARC_TOOLCHAIN := mwdt


  BUILD_ARC_MLI := true
  ARC_MLI_PRE_COMPILED_TARGET := iotdk_arcem9d

ifneq ($(filter no_arc_mli,$(ALL_TAGS)),)
  MLI_LIB_DIR = arc_mli_package
  $(eval $(call add_third_party_download,$(EMBARC_MLI_PRE_COMPILED_URL),$(EMBARC_MLI_PRE_COMPILED_MD5),$(MLI_LIB_DIR),))
else ifeq ($(BUILD_ARC_MLI), true)
  MLI_LIB_DIR = arc_mli_$(ARC_MLI_PRE_COMPILED_TARGET)
endif

  TCF_FILE = $(PWD)/$(MAKEFILE_DIR)/downloads/$(MLI_LIB_DIR)/hw/iotdk_arcem9d.tcf
  LCF_FILE = $(PWD)/$(MAKEFILE_DIR)/targets/arc/iotdk/iotdk.lcf
  UBOOT_FILE := $(PWD)/$(MAKEFILE_DIR)/targets/arc/iotdk/uboot.env
  UBOOT_FILE_NAME := $(notdir $(UBOOT_FILE))

include $(MAKEFILE_DIR)/targets/arc/arc_common.inc

ARC_EXTRA_APP_SETTINGS = \
    BIN_DIR = .$(DLR)\(PS\)bin\n\
    BIN_FILE = $(DLR)\(BIN_DIR\)$(DLR)\(PS\)app.bin\n\
    \nOPT_PERFORMANCE ?= false\n\
    \nifeq \($(DLR)\(OPT_PERFORMANCE\), true\)\
    \nCCFLAGS += -O3 -g\
    \nCXXFLAGS += -O3 -g\
    \nelse\
    \nCCFLAGS += -Os -g\
    \nCXXFLAGS += -Os -g\
    \nendif\n

ARC_EXTRA_APP_RULES = \
     $(DLR)\(BIN_FILE\): $(DLR)\(BIN_DIR\) $(DLR)\(OUT_NAME\)\
     \n\t\elf2bin $(DLR)\(OUT_NAME\) $(DLR)\(BIN_FILE\)\ -b 0x20000000 -t 0x20040000\
     \n\t\@$(DLR)\(CP\) $(UBOOT_FILE_NAME) $(DLR)\(BIN_DIR\)$(DLR)\(PS\)$(UBOOT_FILE_NAME)\
     \n \
     \n$(DLR)\(BIN_DIR\):\
     \n\t\@$(DLR)\(MKDIR\) $(DLR)\(BIN_DIR\)\

ARC_EXTRA_RM_TARGETS = $(DLR)\(BIN_DIR\)

ARC_BIN_DEPEND = $(DLR)\(BIN_DIR\) $(DLR)\(BIN_FILE\)
ARC_BIN_RULE = \t@echo Copy content of $(DLR)\(BIN_DIR\) into the root of SD card and follow instructions

ARC_APP_RUN_CMD = mdb -run -digilent -nooptions $(DLR)\(DBG_ARGS\)
ARC_APP_DEBUG_CMD = mdb -OK -digilent -nooptions $(DLR)\(DBG_ARGS\)
ARC_EXTRA_EXECUTE_RULES = 

MAKE_PROJECT_FILES += $(UBOOT_FILE_NAME)
ARC_TARGET_COPY_FILES += $(notdir $(UBOOT_FILE))!$(UBOOT_FILE)


# Filter out optimization level flag. 
# We set optimization level configurable by OPT_PERFORMANCE variable. 
ifneq (,$(findstring -O3,$(CXXFLAGS)))
  CXXFLAGS := $(filter-out -O%,$(CXXFLAGS))
endif
ifneq (,$(findstring -O3,$(CCFLAGS)))
  CCFLAGS := $(filter-out -O%,$(CCFLAGS))
endif

CXXFLAGS += -Hcrt_initbss
CCFLAGS += -Hcrt_initbss

MICROLITE_CC_HDRS += tensorflow/lite/micro/tools/make/targets/arc/iotdk/flash_programmer/main.c
MICROLITE_CC_HDRS += tensorflow/lite/micro/tools/make/targets/arc/iotdk/flash_programmer/Makefile
MICROLITE_CC_HDRS += tensorflow/lite/micro/tools/make/targets/arc/iotdk/flash_programmer/README.txt


  # for better performance runtime should be built for IoTDK configuration
  # No hostlink library for smaller codesize purpose
  PLATFORM_LDFLAGS += -Hhostlib=

endif
