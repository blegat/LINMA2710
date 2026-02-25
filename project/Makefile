CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDE = -Iinclude

# OpenMP flag: set to empty to disable (e.g., on macOS without OpenMP)
OPENMP_FLAGS ?= -fopenmp

SRC_DIR ?= src

# --- Part 1 & 2: Basic + OpenMP Matrix ---
test_matrix: tests/test_matrix.cpp $(SRC_DIR)/matrix.cpp include/matrix.hpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(INCLUDE) -o test_matrix tests/test_matrix.cpp $(SRC_DIR)/matrix.cpp

run_matrix: test_matrix
	./test_matrix

# --- Part 3: Distributed Matrix (MPI) ---
test_distributed: tests/test_distributed.cpp $(SRC_DIR)/distributed_matrix.cpp $(SRC_DIR)/matrix.cpp include/distributed_matrix.hpp include/matrix.hpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) -o test_distributed tests/test_distributed.cpp $(SRC_DIR)/distributed_matrix.cpp $(SRC_DIR)/matrix.cpp

run_distributed: test_distributed
	mpirun -np 4 ./test_distributed

# --- Part 4: OpenCL Matrix ---
test_opencl: tests/test_opencl.cpp $(SRC_DIR)/matrix_opencl.cpp include/matrix_opencl.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o test_opencl tests/test_opencl.cpp $(SRC_DIR)/matrix_opencl.cpp -lOpenCL

run_opencl: test_opencl
	./test_opencl

# --- Utilities ---
all: test_matrix test_distributed test_opencl

clean:
	rm -f test_matrix test_distributed test_opencl

.PHONY: all clean run_matrix run_distributed run_opencl
