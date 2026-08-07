ifndef CPPC
	CPPC=clang++
endif

CCFLAGS=-O3 -DCL_TARGET_OPENCL_VERSION=300

INC = -I $(CPP_COMMON)

# `-ldl` needed for `dladdr`
LIBS = -lOpenCL -lrt -lpthread -ldl

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -framework OpenCL
endif

TRACY_DIR = ../../../../tracy/public

# For Tracy
CCFLAGS += -g -fno-omit-frame-pointer -rdynamic

OpenCLVectorAdd: OpenCLVectorAdd.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -I $(TRACY_DIR) -o $@

clean:
	rm -f OpenCLVectorAdd
