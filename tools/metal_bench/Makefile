CXX      := clang++
METAL_CPP := ../../external_libs/metal-cpp
CXXFLAGS := -std=c++17 -O3 -Xpreprocessor -fopenmp \
            -I$(METAL_CPP) \
            -Wall -Wno-unused-parameter
LDFLAGS  := -framework Metal -framework Foundation -framework QuartzCore \
            -L$(shell brew --prefix libomp 2>/dev/null)/lib -lomp

main: main.cpp
	$(CXX) $(CXXFLAGS) -I$(shell brew --prefix libomp 2>/dev/null)/include \
	    main.cpp -o main $(LDFLAGS)

clean:
	rm -f main

.PHONY: clean
