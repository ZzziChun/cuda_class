NVCC      := nvcc
CXX       := g++

# 支援多個常見的 compute capability,使用者可在 make 時覆寫
SM_ARCH   ?= sm_75

NVCCFLAGS := -O3 -std=c++14 -arch=$(SM_ARCH) -Iinclude --use_fast_math \
             -Xcompiler "-Wall -Wextra -O3"
CXXFLAGS  := -O3 -std=c++14 -Iinclude -Wall -Wextra

SRC_DIR   := src
BUILD_DIR := build
BIN       := $(BUILD_DIR)/bilateral

CU_SRCS   := $(SRC_DIR)/main.cu $(SRC_DIR)/bilateral_cuda.cu
CPP_SRCS  := $(SRC_DIR)/bilateral_cpu.cpp $(SRC_DIR)/ppm.cpp

CU_OBJS   := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
CPP_OBJS  := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))

.PHONY: all clean run test

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

$(BIN): $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# 產生一張合成測試影像 (1024x1024,含邊緣與雜訊)
data/test.ppm: tools/gen_image.cpp
	$(CXX) $(CXXFLAGS) tools/gen_image.cpp -o $(BUILD_DIR)/gen_image
	$(BUILD_DIR)/gen_image $@ 1024 1024

run: $(BIN) data/test.ppm
	$(BIN) data/test.ppm data/out.ppm 5 3.0 30.0

test: $(BIN) data/test.ppm
	$(BIN) data/test.ppm data/out_small.ppm  3 2.0 25.0
	$(BIN) data/test.ppm data/out_medium.ppm 5 3.0 30.0
	$(BIN) data/test.ppm data/out_large.ppm  7 5.0 40.0

clean:
	rm -rf $(BUILD_DIR) data/out*.ppm
