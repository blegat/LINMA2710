CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O0
TARGET = tests
SRC = mlp_sgd.cpp tests.cpp globals.cpp matrix.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
