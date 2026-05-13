#ifndef BILATERAL_H
#define BILATERAL_H

#include "ppm.h"

// Bilateral filter 參數
//   radius     : 濾波核心半徑 (window size = 2*radius+1)
//   sigma_s    : 空間 (Gaussian) sigma
//   sigma_r    : range (色彩相似度) sigma
struct BFParams {
    int   radius;
    float sigma_s;
    float sigma_r;
};

// CPU baseline
void bilateral_cpu(const Image& in, Image& out, const BFParams& p);

// CUDA naive — 純 global memory 版本
//   回傳值: kernel 執行毫秒數 (不含 memcpy)
float bilateral_cuda_naive(const Image& in, Image& out, const BFParams& p);

// CUDA optimized — shared memory tile + constant memory 空間權重
float bilateral_cuda_shared(const Image& in, Image& out, const BFParams& p);

// 比較兩張影像的平均絕對誤差 (0~255 尺度)
double image_mae(const Image& a, const Image& b);

#endif
