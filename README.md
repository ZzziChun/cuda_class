# CUDA Bilateral Filter

CUDA 平行運算課程期末專題:Bilateral Filter(雙邊濾波)的 CPU / GPU 三版本比較。

實作三個版本並量測效能:

| 版本           | 說明                                                  |
| -------------- | ----------------------------------------------------- |
| **CPU**        | 單執行緒 baseline,作為正確性參考                      |
| **GPU naive**  | 純 global memory 版本                                 |
| **GPU shared** | shared memory tile + constant memory 空間權重(優化版) |

---

## 1. 環境需求

- NVIDIA GPU(支援 CUDA Compute Capability ≥ 5.0)
- CUDA Toolkit(內含 `nvcc`)
- `g++`(支援 C++14)
- Linux / WSL

預設架構為 `sm_75`(Turing,例如 RTX 20 系列、T4)。若你的 GPU 不同,在 `make` 時用 `SM_ARCH` 覆寫,例如:

```bash
make SM_ARCH=sm_86   # Ampere (RTX 30 系列)
make SM_ARCH=sm_89   # Ada    (RTX 40 系列)
make SM_ARCH=sm_60   # Pascal (P100)
```

---

## 2. 目錄結構

```
cuda_class/
├── Makefile
├── include/
│   ├── bilateral.h        # API 與參數定義
│   └── ppm.h              # PPM 讀寫
├── src/
│   ├── main.cu            # Driver:跑三個版本 + 印效能
│   ├── bilateral_cpu.cpp  # CPU baseline
│   ├── bilateral_cuda.cu  # GPU naive + shared 版本
│   └── ppm.cpp
├── tools/
│   ├── gen_image.cpp      # 產生合成測試影像
│   └── cpu_only.cpp       # 只跑 CPU 版本的小工具
├── data/                  # 測試影像與輸出
└── build/                 # 編譯產物
```

---

## 3. 編譯

```bash
make                       # 編譯主程式 build/bilateral
make SM_ARCH=sm_86         # 指定 GPU 架構
make clean                 # 清除 build/ 與 data/out*.ppm
```

---

## 4. 快速開始(一鍵跑)

```bash
make run
```

這會:

1. 編譯主程式
2. 產生 1024×1024 的測試影像 `data/test.ppm`
3. 以參數 `radius=5, sigma_s=3.0, sigma_r=30.0` 跑三個版本
4. 結果寫到 `data/out.ppm`

跑一組常用的小 / 中 / 大濾波核:

```bash
make test
```

---

## 5. 直接執行主程式

```bash
./build/bilateral <input.ppm> <output.ppm> [radius] [sigma_s] [sigma_r]
```

| 參數         | 說明                           | 預設   |
| ------------ | ------------------------------ | ------ |
| `input.ppm`  | 輸入影像(P6 PPM 格式)          | —      |
| `output.ppm` | 輸出影像(shared 版結果)        | —      |
| `radius`     | 核心半徑,window = `2*radius+1` | `5`    |
| `sigma_s`    | 空間 Gaussian sigma            | `3.0`  |
| `sigma_r`    | range(色彩相似度)sigma         | `30.0` |

範例:

```bash
./build/bilateral data/test.ppm data/out.ppm 7 5.0 40.0
```

執行後會印出 GPU 資訊、影像大小、各版本時間、speedup 與 MAE(對 CPU 的平均絕對誤差,正確性指標):

```
[GPU] NVIDIA ... (CC 7.5, 40 SMs, ...)
[Image]   1024 x 1024  (3 channels)
[Params]  radius=5  sigma_s=3.00  sigma_r=30.00  (window=11x11)
-----------------------------------------------------------
[CPU         ]      xxxx.xx ms
[GPU naive   ]        xx.xx ms   speedup vs CPU = ...x   MAE vs CPU = ...
[GPU shared  ]         x.xx ms   speedup vs CPU = ...x   MAE vs CPU = ...
-----------------------------------------------------------
[Shared vs Naive speedup]  ...x
```

---

## 6. 產生自訂測試影像

```bash
make data/test.ppm                              # 預設 1024×1024
./build/gen_image data/test_2048.ppm 2048 2048  # 自訂大小
```

合成影像包含色塊邊緣、中央亮環與高斯雜訊,適合驗證 bilateral 的「保邊去雜訊」效果。

也可以用任何 P6 PPM 影像當輸入。常見影像轉成 PPM:

```bash
# 用 ImageMagick
convert input.jpg -compress none data/input.ppm
```

---

## 7. 輸入 / 輸出格式

- **格式**:P6 PPM(二進位,8-bit per channel)
- **通道**:RGB(3 通道)
- 灰階圖請先轉成 RGB PPM。

檢視輸出:任何看 PPM 的工具都可以,例如 `eog data/out.ppm`、VS Code PPM 預覽外掛、或轉成 PNG:

```bash
convert data/out.ppm data/out.png
```

---

## 8. 參數調整建議

| 想要的效果           | 建議                                |
| -------------------- | ----------------------------------- |
| 輕度去雜訊、保留細節 | `radius=3, sigma_s=2.0, sigma_r=25` |
| 中等去雜訊           | `radius=5, sigma_s=3.0, sigma_r=30` |
| 強去雜訊、明顯卡通化 | `radius=7, sigma_s=5.0, sigma_r=40` |

- `radius` 越大 → window 越大,GPU shared 版相對於 naive 的加速越明顯。
- `sigma_r` 越大 → 越不在意色彩差異,結果越平滑。
- `sigma_s` 越大 → 空間影響範圍越廣,通常配合較大的 `radius`。

---

## 9. 常見問題

- **`nvcc: command not found`**:沒安裝 CUDA Toolkit 或沒加進 `PATH`。
- **執行時 `no kernel image is available for execution on the device`**:`SM_ARCH` 跟實際 GPU 不合,改用對應的 `sm_XX` 重新 `make`。
- **MAE 很大**:檢查 `radius` 是否過大導致 shared memory tile 超過上限,或參數型別轉換問題。正常情況下三個版本的 MAE 應 < 1.0。
