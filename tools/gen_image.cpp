// 產生合成測試影像 (P6 PPM):有色塊邊緣 + 高斯雜訊,
// 用來驗證 bilateral filter 的 "保邊去雜訊" 效果。

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <random>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <out.ppm> <W> <H>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int W = atoi(argv[2]);
    int H = atoi(argv[3]);

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 18.0f);

    uint8_t* buf = (uint8_t*)malloc((size_t)W * H * 3);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // 幾個色塊 + 環形 + 對角線
            int region = (x / (W / 4)) + (y / (H / 4)) * 4;
            float r, g, b;
            switch (region % 6) {
                case 0: r = 220; g =  40; b =  40; break;
                case 1: r =  40; g = 200; b =  60; break;
                case 2: r =  50; g =  70; b = 220; break;
                case 3: r = 230; g = 200; b =  50; break;
                case 4: r =  60; g = 200; b = 210; break;
                default:r = 200; g = 100; b = 200; break;
            }
            // 中央亮環
            float cx = W * 0.5f, cy = H * 0.5f;
            float d = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (d > W * 0.30f && d < W * 0.32f) { r = 250; g = 250; b = 250; }
            // 加雜訊
            r += noise(rng); g += noise(rng); b += noise(rng);
            if (r < 0) r = 0; if (r > 255) r = 255;
            if (g < 0) g = 0; if (g > 255) g = 255;
            if (b < 0) b = 0; if (b > 255) b = 255;

            int i = (y * W + x) * 3;
            buf[i + 0] = (uint8_t)r;
            buf[i + 1] = (uint8_t)g;
            buf[i + 2] = (uint8_t)b;
        }
    }

    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen"); return 2; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(buf, 1, (size_t)W * H * 3, f);
    fclose(f);
    free(buf);
    printf("Generated %s (%d x %d)\n", path, W, H);
    return 0;
}
