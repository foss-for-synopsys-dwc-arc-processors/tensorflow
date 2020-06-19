ifeq ($(TARGET), arc_iotdk)

# Patch of arc make project to adjust it specifically 
# for micro speech example. In particular:
# - Use Linker command file with better usage of fast memory and tweak scratch buffers
# - Stripout TFLM reference code by default.
# - set optimization speed by default.

  MS_EXAMPLE_PATH = tensorflow/lite/micro/examples/micro_speech
  
  MICRO_SPEECH_HDRS += \
  micro_speech_patch.txt
  
  MICRO_SPEECH_TEST_HDRS += \
  micro_speech_patch.txt
  
  MICRO_SPEECH_MOCK_HDRS += \
  micro_speech_patch.txt



%/micro_speech_patch.txt: %/iotdk.lcf %/Makefile %/uboot.env
	@cp $(MS_EXAMPLE_PATH)/arc_iotdk/iotdk.lcf $< 
	@echo iotdk.lcf: use example specific lcf > $@

	@cp $(MS_EXAMPLE_PATH)/arc_iotdk/uboot.env $(word 3, $^) 
	@echo uboot.env: use example specific uboot environment > $@
	
	@sed -E -i 's#-Hheap=[^ ]*#\-Hheap=16K \-Hstack=16K#g' $(word 2, $^)
	@sed -E -i 's#OPT_PERFORMANCE *\?= *false#OPT_PERFORMANCE \?= true#' $(word 2, $^)
	@sed -E -i 's#MLI_ONLY *\?= *false#MLI_ONLY \?= true\n\
	CXXFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0\
	CCFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0#'\
 	  $(word 2, $^)
	@echo Makefile: Set exact heap/stack size >> $@
	@echo Makefile: Switch default optimization to performance \(-O3\) >> $@
	@echo Makefile: set scratch buffers size to 0 >> $@
	@echo Makefile: No Reference fallback for MLI supported functions >> $@
	@echo Makefile: MLI_ONLY is true >> $@

	@sed -i '/^BIN_DIR/s/$$/\nDATA_BIN_FILE = $$\(BIN_DIR\)$$\(PS\)data.bin/' $(word 2, $^)
	@sed -E -i '/^.\(BIN_FILE\):/s/$$/\n\telf2bin $$\(OUT_NAME\) $$\(DATA_BIN_FILE\) -b 0xE0000000 -t 0xE0008000/' $(word 2, $^)


endif
