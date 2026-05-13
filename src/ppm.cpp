#include "ppm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

Image image_alloc(int width, int height, int channels) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data = (uint8_t*)malloc((size_t)width * height * channels);
    return img;
}

void image_free(Image& img) {
    if (img.data) free(img.data);
    img.data = nullptr;
    img.width = img.height = img.channels = 0;
}

// 跳過 PPM 標頭中的註解與空白
static int skip_ws_and_comments(FILE* f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n') {}
        } else if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            return c;
        }
    }
    return EOF;
}

static bool read_int(FILE* f, int& out) {
    int c = skip_ws_and_comments(f);
    if (c == EOF) return false;
    int val = 0;
    if (c < '0' || c > '9') return false;
    do {
        val = val * 10 + (c - '0');
        c = fgetc(f);
    } while (c >= '0' && c <= '9');
    out = val;
    return true;
}

bool ppm_load(const char* path, Image& img) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ppm_load: cannot open %s\n", path);
        return false;
    }

    char magic[3] = {0};
    if (fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || magic[1] != '6') {
        fprintf(stderr, "ppm_load: not a P6 PPM\n");
        fclose(f);
        return false;
    }

    int w, h, maxval;
    if (!read_int(f, w) || !read_int(f, h) || !read_int(f, maxval)) {
        fprintf(stderr, "ppm_load: header parse error\n");
        fclose(f);
        return false;
    }
    if (maxval != 255) {
        fprintf(stderr, "ppm_load: only maxval 255 supported (got %d)\n", maxval);
        fclose(f);
        return false;
    }
    // 注意:read_int 已經吃掉 maxval 後的單一空白分隔字元

    img = image_alloc(w, h, 3);
    size_t n = (size_t)w * h * 3;
    if (fread(img.data, 1, n, f) != n) {
        fprintf(stderr, "ppm_load: pixel data short read\n");
        image_free(img);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

bool ppm_save(const char* path, const Image& img) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ppm_save: cannot open %s\n", path);
        return false;
    }
    fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    size_t n = (size_t)img.width * img.height * 3;
    bool ok = fwrite(img.data, 1, n, f) == n;
    fclose(f);
    return ok;
}
