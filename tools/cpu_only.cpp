// 只跑 CPU 版本,並輸出結果影像 (driver 不在時用來確認正確性)
#include "bilateral.h"
#include "ppm.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s in.ppm out.ppm [r] [ss] [sr]\n", argv[0]); return 1; }
    BFParams p;
    p.radius  = argc > 3 ? atoi(argv[3]) : 5;
    p.sigma_s = argc > 4 ? atof(argv[4]) : 3.0f;
    p.sigma_r = argc > 5 ? atof(argv[5]) : 30.0f;
    Image in{}, out{};
    if (!ppm_load(argv[1], in)) return 2;
    auto t0 = std::chrono::high_resolution_clock::now();
    bilateral_cpu(in, out, p);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("CPU: %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
    ppm_save(argv[2], out);
    return 0;
}
