#include "bilateral.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// CPU 參考實作 — 直接照公式算,作為正確性與效能基準
void bilateral_cpu(const Image& in, Image& out, const BFParams& p) {
    const int W = in.width, H = in.height;
    const int R = p.radius;
    const float inv_two_ss2 = 1.0f / (2.0f * p.sigma_s * p.sigma_s);
    const float inv_two_sr2 = 1.0f / (2.0f * p.sigma_r * p.sigma_r);

    out = image_alloc(W, H, 3);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int ci = (y * W + x) * 3;
            const float cr = in.data[ci + 0];
            const float cg = in.data[ci + 1];
            const float cb = in.data[ci + 2];

            float wsum = 0.0f;
            float sr = 0.0f, sg = 0.0f, sb = 0.0f;

            for (int dy = -R; dy <= R; ++dy) {
                int ny = y + dy;
                if (ny < 0) ny = 0;
                else if (ny >= H) ny = H - 1;
                for (int dx = -R; dx <= R; ++dx) {
                    int nx = x + dx;
                    if (nx < 0) nx = 0;
                    else if (nx >= W) nx = W - 1;

                    const int ni = (ny * W + nx) * 3;
                    const float r = in.data[ni + 0];
                    const float g = in.data[ni + 1];
                    const float b = in.data[ni + 2];

                    const float spatial = (float)(dx * dx + dy * dy) * inv_two_ss2;
                    const float dr = r - cr, dg = g - cg, db = b - cb;
                    const float range = (dr*dr + dg*dg + db*db) * inv_two_sr2;
                    const float w = expf(-(spatial + range));

                    wsum += w;
                    sr += w * r;
                    sg += w * g;
                    sb += w * b;
                }
            }

            const float inv = 1.0f / wsum;
            out.data[ci + 0] = (uint8_t)std::min(255.0f, std::max(0.0f, sr * inv));
            out.data[ci + 1] = (uint8_t)std::min(255.0f, std::max(0.0f, sg * inv));
            out.data[ci + 2] = (uint8_t)std::min(255.0f, std::max(0.0f, sb * inv));
        }
    }
}

double image_mae(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
        return -1.0;
    }
    const size_t n = (size_t)a.width * a.height * a.channels;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += std::abs((int)a.data[i] - (int)b.data[i]);
    }
    return sum / (double)n;
}
