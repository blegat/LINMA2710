CXX = clang++
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra
TARGET = sparse_matrix_csc
SOURCES = main.cpp
HEADERS = sparse_matrix_csc.hpp

# Default target
all: $(TARGET)

# Compile the executable
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Run the program
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
