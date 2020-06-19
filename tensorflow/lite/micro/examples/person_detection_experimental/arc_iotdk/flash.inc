
FLASH_PROGRAMMER_ROOT := ./tensorflow/lite/micro/tools/make/targets/arc/iotdk/flash_programmer

flash_programmer.elf:
	$(MAKE) -f $(FLASH_PROGRAMMER_ROOT)/Makefile IOTDK_TCF=iotdk_arcem9d.tcf  IOTDK_LCF=iotdk.lcf all


protobuf.bin: $(OUT_NAME)
	elf2bin $(OUT_NAME) $@ -sn .protobuf

flash_iotdk: $(OUT_NAME) flash_programmer.elf protobuf.bin
	$(APP_RUN) flash_programmer.elf protobuf.bin 0x80000
	@echo !
	@echo ! PLEASE RESET THE BOARD 
	@echo !
