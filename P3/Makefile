CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O0
TARGET = main
SRC = matrix_opencl.cpp main.cpp mlp_sgd.cpp globals.cpp

all:
	$(MAKE) clean && $(MAKE) run

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -lOpenCL

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
