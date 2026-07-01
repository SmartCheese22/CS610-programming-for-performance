# Makefile for problem3.cpp

CXX = g++
CXXFLAGS = -std=c++17 -masm=att -msse4 -mavx2 -march=native -fopenmp -fverbose-asm \
           -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -fcf-protection=none -O2
OUTPUT = problem3.out

all: $(OUTPUT)

$(OUTPUT): problem3.cpp
	$(CXX) $(CXXFLAGS) problem3.cpp -o $(OUTPUT) -pthread

clean:
	rm -f $(OUTPUT)
