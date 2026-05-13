#include "bilateral.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _err = (call);                                             \
    if (_err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_err));             \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// 編譯期上限 — 半徑 ≤ 16,window ≤ 33×33 = 1089
#define MAX_RADIUS 16
#define MAX_W_SIZE ((MAX_RADIUS) * 2 + 1)

// constant memory:預先算好的空間 Gaussian 權重 (1D),長度 = window size
__constant__ float c_spatial_weights[MAX_W_SIZE * MAX_W_SIZE];

// 邊界 clamp helper
__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ============================================================
// Kernel 1: Naive — 每 thread 算一個像素,鄰居皆從 global memory 拿
// ============================================================
__global__ void bilateral_naive_kernel(const uint8_t* __restrict__ in,
                                       uint8_t* __restrict__ out,
                                       int W, int H, int R,
                                       float inv_two_ss2,
                                       float inv_two_sr2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int ci = (y * W + x) * 3;
    const float cr = in[ci + 0];
    const float cg = in[ci + 1];
    const float cb = in[ci + 2];

    float wsum = 0.0f, sr = 0.0f, sg = 0.0f, sb = 0.0f;

    for (int dy = -R; dy <= R; ++dy) {
        const int ny = clamp_int(y + dy, 0, H - 1);
        for (int dx = -R; dx <= R; ++dx) {
            const int nx = clamp_int(x + dx, 0, W - 1);
            const int ni = (ny * W + nx) * 3;
            const float r = in[ni + 0];
            const float g = in[ni + 1];
            const float b = in[ni + 2];

            const float spatial = (float)(dx * dx + dy * dy) * inv_two_ss2;
            const float dr = r - cr, dg = g - cg, db = b - cb;
            const float range = (dr*dr + dg*dg + db*db) * inv_two_sr2;
            const float w = __expf(-(spatial + range));

            wsum += w;
            sr += w * r;
            sg += w * g;
            sb += w * b;
        }
    }

    const float inv = 1.0f / wsum;
    out[ci + 0] = (uint8_t)fminf(255.0f, fmaxf(0.0f, sr * inv));
    out[ci + 1] = (uint8_t)fminf(255.0f, fmaxf(0.0f, sg * inv));
    out[ci + 2] = (uint8_t)fminf(255.0f, fmaxf(0.0f, sb * inv));
}

// ============================================================
// Kernel 2: Shared memory tile + constant memory 空間權重
//   - 每個 block 處理 TILE x TILE 個像素
//   - 把 (TILE + 2R) x (TILE + 2R) 的 halo 載到 shared memory
//   - 空間權重從 __constant__ 拿,免在 kernel 重算
// ============================================================
template <int TILE>
__global__ void bilateral_shared_kernel(const uint8_t* __restrict__ in,
                                        uint8_t* __restrict__ out,
                                        int W, int H, int R,
                                        float inv_two_sr2) {
    extern __shared__ uint8_t s_mem[];     // 大小由 host 端決定
    const int SW = TILE + 2 * R;           // shared tile 寬高

    const int gx0 = blockIdx.x * TILE;
    const int gy0 = blockIdx.y * TILE;

    // 協作載入 — 每個 thread 可能負責多個 shared 位置
    for (int sy = threadIdx.y; sy < SW; sy += blockDim.y) {
        for (int sx = threadIdx.x; sx < SW; sx += blockDim.x) {
            int gx = clamp_int(gx0 + sx - R, 0, W - 1);
            int gy = clamp_int(gy0 + sy - R, 0, H - 1);
            int gi = (gy * W + gx) * 3;
            int si = (sy * SW + sx) * 3;
            s_mem[si + 0] = in[gi + 0];
            s_mem[si + 1] = in[gi + 1];
            s_mem[si + 2] = in[gi + 2];
        }
    }
    __syncthreads();

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int gx = gx0 + lx;
    const int gy = gy0 + ly;
    if (gx >= W || gy >= H || lx >= TILE || ly >= TILE) return;

    const int sx_c = lx + R;
    const int sy_c = ly + R;
    const int sci = (sy_c * SW + sx_c) * 3;
    const float cr = s_mem[sci + 0];
    const float cg = s_mem[sci + 1];
    const float cb = s_mem[sci + 2];

    const int WSIZE = 2 * R + 1;
    float wsum = 0.0f, sr = 0.0f, sg = 0.0f, sb = 0.0f;

    #pragma unroll 1
    for (int dy = -R; dy <= R; ++dy) {
        #pragma unroll 1
        for (int dx = -R; dx <= R; ++dx) {
            const int si = ((sy_c + dy) * SW + (sx_c + dx)) * 3;
            const float r = s_mem[si + 0];
            const float g = s_mem[si + 1];
            const float b = s_mem[si + 2];

            const float spatial_w = c_spatial_weights[(dy + R) * WSIZE + (dx + R)];
            const float dr = r - cr, dg = g - cg, db = b - cb;
            const float range = (dr*dr + dg*dg + db*db) * inv_two_sr2;
            const float w = spatial_w * __expf(-range);

            wsum += w;
            sr += w * r;
            sg += w * g;
            sb += w * b;
        }
    }

    const float inv = 1.0f / wsum;
    const int gi = (gy * W + gx) * 3;
    out[gi + 0] = (uint8_t)fminf(255.0f, fmaxf(0.0f, sr * inv));
    out[gi + 1] = (uint8_t)fminf(255.0f, fmaxf(0.0f, sg * inv));
    out[gi + 2] = (uint8_t)fminf(255.0f, fmaxf(0.0f, sb * inv));
}

// ============================================================
// Host wrappers
// ============================================================

static void upload_spatial_weights(int R, float sigma_s) {
    const int WSIZE = 2 * R + 1;
    float h_w[MAX_W_SIZE * MAX_W_SIZE] = {0};
    const float inv_two_ss2 = 1.0f / (2.0f * sigma_s * sigma_s);
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            h_w[(dy + R) * WSIZE + (dx + R)] =
                expf(-(float)(dx*dx + dy*dy) * inv_two_ss2);
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_spatial_weights, h_w,
                                  sizeof(float) * WSIZE * WSIZE));
}

float bilateral_cuda_naive(const Image& in, Image& out, const BFParams& p) {
    if (p.radius > MAX_RADIUS) {
        fprintf(stderr, "radius %d exceeds MAX_RADIUS=%d\n", p.radius, MAX_RADIUS);
        exit(EXIT_FAILURE);
    }
    out = image_alloc(in.width, in.height, 3);
    const size_t bytes = (size_t)in.width * in.height * 3;

    uint8_t *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in.data, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((in.width  + block.x - 1) / block.x,
              (in.height + block.y - 1) / block.y);

    const float inv_two_ss2 = 1.0f / (2.0f * p.sigma_s * p.sigma_s);
    const float inv_two_sr2 = 1.0f / (2.0f * p.sigma_r * p.sigma_r);

    cudaEvent_t e_start, e_stop;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_stop));
    CUDA_CHECK(cudaEventRecord(e_start));

    bilateral_naive_kernel<<<grid, block>>>(d_in, d_out, in.width, in.height,
                                            p.radius, inv_two_ss2, inv_two_sr2);

    CUDA_CHECK(cudaEventRecord(e_stop));
    CUDA_CHECK(cudaEventSynchronize(e_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));

    CUDA_CHECK(cudaMemcpy(out.data, d_out, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(e_start); cudaEventDestroy(e_stop);
    return ms;
}

float bilateral_cuda_shared(const Image& in, Image& out, const BFParams& p) {
    if (p.radius > MAX_RADIUS) {
        fprintf(stderr, "radius %d exceeds MAX_RADIUS=%d\n", p.radius, MAX_RADIUS);
        exit(EXIT_FAILURE);
    }
    out = image_alloc(in.width, in.height, 3);
    const size_t bytes = (size_t)in.width * in.height * 3;

    uint8_t *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in.data, bytes, cudaMemcpyHostToDevice));

    upload_spatial_weights(p.radius, p.sigma_s);

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((in.width  + TILE - 1) / TILE,
              (in.height + TILE - 1) / TILE);
    const int SW = TILE + 2 * p.radius;
    const size_t shmem_bytes = (size_t)SW * SW * 3 * sizeof(uint8_t);

    const float inv_two_sr2 = 1.0f / (2.0f * p.sigma_r * p.sigma_r);

    cudaEvent_t e_start, e_stop;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_stop));
    CUDA_CHECK(cudaEventRecord(e_start));

    bilateral_shared_kernel<TILE><<<grid, block, shmem_bytes>>>(
        d_in, d_out, in.width, in.height, p.radius, inv_two_sr2);

    CUDA_CHECK(cudaEventRecord(e_stop));
    CUDA_CHECK(cudaEventSynchronize(e_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));

    CUDA_CHECK(cudaMemcpy(out.data, d_out, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(e_start); cudaEventDestroy(e_stop);
    return ms;
}
