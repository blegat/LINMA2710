CXX = mpic++
CXXFLAGS = -std=c++17 -Wall -Wextra -O0
TARGET = distributedtests
OBJ = matrix.o distributedmatrix.o distributedtests.o mlp_sgd_distributed.o globals.o
HEADERS = abstractmatrix.hpp matrix.hpp distributedmatrix.hpp globals.hpp

all:
	$(MAKE) clean && $(MAKE) run

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

matrix.o: matrix.cpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp

distributedmatrix.o: distributedmatrix.cpp distributedmatrix.hpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c distributedmatrix.cpp

distributedtests.o: distributedtests.cpp distributedmatrix.hpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c distributedtests.cpp

mlp_sgd_distributed.o: mlp_sgd_distributed.cpp globals.hpp abstractmatrix.hpp matrix.hpp distributedmatrix.hpp
	$(CXX) $(CXXFLAGS) -c mlp_sgd_distributed.cpp

globals.o: globals.cpp globals.hpp mlp_sgd_distributed.cpp
	$(CXX) $(CXXFLAGS) -c globals.cpp

run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all run clean
