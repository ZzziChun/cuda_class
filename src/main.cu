// CUDA Bilateral Filter — benchmark / driver
//
// 用法:
//   bilateral <input.ppm> <output.ppm> [radius=5] [sigma_s=3.0] [sigma_r=30.0]
//
// 會跑三個版本 (CPU / GPU naive / GPU shared),印出時間與 MAE,
// 並把 shared 版本的結果寫到 output.ppm。

#include "bilateral.h"
#include "ppm.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static void print_gpu_info() {
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        printf("[GPU] (info unavailable)\n");
        return;
    }
    printf("[GPU] %s  (CC %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <in.ppm> <out.ppm> [radius=5] [sigma_s=3.0] [sigma_r=30.0]\n",
                argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    BFParams p;
    p.radius  = (argc > 3) ? atoi(argv[3]) : 5;
    p.sigma_s = (argc > 4) ? atof(argv[4]) : 3.0f;
    p.sigma_r = (argc > 5) ? atof(argv[5]) : 30.0f;

    Image in{};
    if (!ppm_load(in_path, in)) return 2;

    printf("===========================================================\n");
    printf(" Bilateral Filter — CUDA Parallel Computing Final Project\n");
    printf("===========================================================\n");
    print_gpu_info();
    printf("[Image]   %d x %d  (%d channels)\n", in.width, in.height, in.channels);
    printf("[Params]  radius=%d  sigma_s=%.2f  sigma_r=%.2f  (window=%dx%d)\n",
           p.radius, p.sigma_s, p.sigma_r,
           2 * p.radius + 1, 2 * p.radius + 1);
    printf("-----------------------------------------------------------\n");

    Image out_cpu{}, out_gpu_naive{}, out_gpu_shared{};

    // --- CPU baseline ---
    auto t0 = std::chrono::high_resolution_clock::now();
    bilateral_cpu(in, out_cpu, p);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[CPU         ]  %10.2f ms\n", cpu_ms);

    // --- GPU naive ---
    // warm-up + 取多次平均更穩,但這裡只跑一次代表
    float ms_naive = bilateral_cuda_naive(in, out_gpu_naive, p);
    printf("[GPU naive   ]  %10.2f ms   speedup vs CPU = %6.2fx   MAE vs CPU = %.3f\n",
           ms_naive, cpu_ms / ms_naive,
           image_mae(out_cpu, out_gpu_naive));

    // --- GPU shared ---
    float ms_shared = bilateral_cuda_shared(in, out_gpu_shared, p);
    printf("[GPU shared  ]  %10.2f ms   speedup vs CPU = %6.2fx   MAE vs CPU = %.3f\n",
           ms_shared, cpu_ms / ms_shared,
           image_mae(out_cpu, out_gpu_shared));

    printf("-----------------------------------------------------------\n");
    printf("[Shared vs Naive speedup]  %.2fx\n", ms_naive / ms_shared);
    printf("===========================================================\n");

    ppm_save(out_path, out_gpu_shared);
    printf("Wrote: %s\n", out_path);

    image_free(in);
    image_free(out_cpu);
    image_free(out_gpu_naive);
    image_free(out_gpu_shared);
    return 0;
}
