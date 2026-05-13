#ifndef PPM_H
#define PPM_H

#include <cstdint>
#include <cstddef>

struct Image {
    int width;
    int height;
    int channels;          // 3 = RGB
    uint8_t* data;         // size = width * height * channels
};

// 讀取 PPM (P6) — 失敗回傳 false
bool ppm_load(const char* path, Image& img);

// 寫出 PPM (P6) — 失敗回傳 false
bool ppm_save(const char* path, const Image& img);

void image_free(Image& img);
Image image_alloc(int width, int height, int channels = 3);

#endif
