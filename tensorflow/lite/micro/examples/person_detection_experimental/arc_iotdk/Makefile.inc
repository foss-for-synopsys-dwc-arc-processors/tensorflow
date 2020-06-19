ifeq ($(TARGET), arc_iotdk)

# Patch of arc make project to adjust it specifically for iotdk
# for experimental person detection example. In particular:
# - Use Example specific Linker command file (see ./iotdk.lcf beside this makefile)
# - Stripout TFLM reference code by default. Set default optimization to speed
# - Add rules to write data into SPI flash
# - Tweak tensor arena size to fit CCMs

  PD_EXAMPLE_PATH = tensorflow/lite/micro/examples/person_detection_experimental
  
  person_detection_HDRS += \
  person_detection_int8_patch.txt \
  
  person_detection_TEST_HDRS += \
  person_detection_int8_test_patch.txt \


%/temp_log.txt: %/iotdk.lcf %/Makefile
	@cp $(PD_EXAMPLE_PATH)/arc_iotdk/iotdk.lcf $< 
	@echo iotdk.lcf: use example specific lcf > $@
	
	@sed -E -i 's#MLI_ONLY *\?= *false#MLI_ONLY \?= true#' $(word 2, $^)
	@sed -E -i 's#OPT_PERFORMANCE *\?= *false#OPT_PERFORMANCE \?= true#' $(word 2, $^)
	@echo Makefile: MLI_ONLY is true >> $@
	@echo Makefile: Switch default optimization to performance \(-O3\) >> $@

	@sed -i '/^run:/s/$$/\n\t@echo !\n\t@echo MAKE SURE YOU WROTE MODEL INTO FLASH BY \"flash_iotdk\" MAKE RULE BEFORE RUN\n\t@echo !/' $(word 2, $^)
	@sed -i '/^debug:/s/$$/\n\t@echo !\n\t@echo MAKE SURE YOU WROTE MODEL INTO FLASH BY \"flash_iotdk\" MAKE RULE BEFORE RUN\n\t@echo !/' $(word 2, $^)
	@sed -i '/^app:/s/$$/\n\t@echo !\n\t@echo MAKE SURE YOU WROTE MODEL INTO FLASH BY \"flash_iotdk\" MAKE RULE BEFORE RUN\n\t@echo !/' $(word 2, $^)
	@cat  $(PD_EXAMPLE_PATH)/arc_iotdk/flash.inc >> $(word 2, $^)
	@echo Makefile: Add model flashing rules >> $@


%/person_detection_int8_patch.txt: %/temp_log.txt %/$(PD_EXAMPLE_PATH)/main_functions.cc 
	@mv $< $@

	@sed -E -i 's#kTensorArenaSize = 136#kTensorArenaSize = 125#' $(word 2, $^)
	@echo $(word 2, $^): Reduce tensor arena size to bare minimum \(to fit onchip memory\)>> $@


%/person_detection_int8_test_patch.txt: %/temp_log.txt %/$(PD_EXAMPLE_PATH)/person_detection_test.cc 
	@mv $< $@

	@sed -E -i 's#tensor_arena_size = 136#tensor_arena_size = 125#' $(word 2, $^)
	@echo $(word 2, $^): Reduce tensor arena size to bare minimum \(to fit onchip memory\)>> $@

endif
