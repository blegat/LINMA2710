ifndef CC
	CC = clang
endif

CCFLAGS=-O3 -lm -DCL_TARGET_OPENCL_VERSION=300

LIBS = -lOpenCL -fopenmp

COMMON_DIR = ../C_common

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL 
endif

CCFLAGS += -DDEVICE=$(DEVICE)

vadd_chain: vadd_chain.c $(COMMON_DIR)/device_info.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@

clean:
	rm -f vadd_chain
