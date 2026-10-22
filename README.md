# Panorama Image Stitching

Đồ án xây dựng chương trình ghép hai ảnh có vùng chồng lấp thành một ảnh toàn cảnh (panorama), bao gồm phiên bản tuần tự trên CPU và các phiên bản tăng tốc bằng CUDA trên GPU.

---

## 1. Yêu cầu chuẩn bị dữ liệu (Dataset)

Trước khi đưa vào pipeline ghép ảnh, các ảnh chụp thực tế cần được chuẩn hóa:
- **Cắt tỉ lệ 16:9 ở giữa (Center-crop):** Giữ đúng phối cảnh trung tâm, tránh méo hình.
- **Resize về 3 độ phân giải:**
  - `small`: 1280 × 720 (dùng để test và debug nhanh)
  - `medium`: 1920 × 1080 (chuẩn đo đạc chính)
  - `large`: 3840 × 2160 (4K, kiểm tra hiệu năng mở rộng của GPU)

---

## 2. Cài đặt thư viện

Công cụ tiền xử lý yêu cầu Python 3, thư viện `Pillow` và `pi-heif` (để đọc ảnh `.HEIC` chụp từ iPhone):

```bash
pip install -r tools/requirements.txt
```

---

## 3. Chuyển ảnh HEIC sang PNG (`tools/convert_images.py`)

Ảnh gốc trong `datasets/` ở định dạng `.HEIC`. Công cụ quét đệ quy toàn bộ thư mục, chuyển mọi ảnh `.HEIC`/`.HEIF` sang `.png` (không mất dữ liệu) **ở độ phân giải gốc, không crop, không resize**, và giữ nguyên cấu trúc thư mục con. Khi đọc ảnh, công cụ tự xoay ảnh theo EXIF orientation và chuyển hệ màu Display P3 sang sRGB.

```bash
# Mặc định: datasets/ -> datasets_png/
python tools/convert_images.py

# Chỉ định thư mục vào/ra khác
python tools/convert_images.py -i raw_photos -o converted

# Chuyển lại cả những ảnh đã có PNG
python tools/convert_images.py --overwrite
```

Ví dụ: `datasets/standard pairs/Indoor/Pair 1/IMG_1147.HEIC` → `datasets_png/standard pairs/Indoor/Pair 1/IMG_1147.png`.

---

## 4. Resize về 3 độ phân giải (`tools/resize_images.py`)

Công cụ hỗ trợ các định dạng: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.heic`, `.heif` (loại bỏ các file lạ như `.pnm`). Mỗi ảnh được xoay theo EXIF, chuyển sang sRGB, center-crop 16:9 rồi resize bằng Lanczos. Ảnh chỉ được đọc và crop một lần cho cả 3 độ phân giải.

### 4.1. Resize toàn bộ dataset
```bash
python tools/resize_images.py -i datasets_png -o datasets_resized
```
Lệnh trên quét đệ quy `datasets_png/` và xuất ảnh `.png` vào `datasets_resized/`, giữ nguyên cấu trúc thư mục con bên trong mỗi độ phân giải:
```
datasets_resized/
├── small/   (1280×720)   standard pairs/Indoor/Pair 1/IMG_1147.png, ...
├── medium/  (1920×1080)  ...
└── large/   (3840×2160)  ...
```
Ảnh đã có đầu ra sẽ được bỏ qua; thêm `--overwrite` để tạo lại.

### 4.2. Resize 1 ảnh hoặc thư mục khác
```bash
# 1 ảnh ra cả 3 kích thước -> dataset/{small,medium,large}/duong_dan_anh.png
python tools/resize_images.py -i duong_dan_anh.png -o dataset

# Chỉ 1 kích thước
python tools/resize_images.py -i duong_dan_anh.png -o dataset -r medium

# Thư mục khác (quét đệ quy)
python tools/resize_images.py -i raw_photos -o dataset
```

### 4.3. Các tùy chọn khác
- **Độ phân giải:** `-r small|medium|large|all` (mặc định `all`) hoặc tự chọn `-r 800x600`
- **Định dạng đầu ra:** `-f png` (mặc định, không mất dữ liệu) hoặc `-f jpg` (quality 95, nhẹ hơn nhiều)
- **Không crop 16:9 (chỉ resize trực tiếp):** thêm `--no-crop`
- **Tạo lại ảnh đã có:** thêm `--overwrite`

---

## 5. Cấu trúc mã nguồn (`src/`)

| File | Nội dung |
|------|----------|
| `main.cu` | Đọc tham số dòng lệnh, đọc ảnh, chạy 1 lần warm-up + N lần đo, in bảng thời gian, lưu panorama |
| `versions.h` | Khai báo `run_v0`, `run_v1`, … (mỗi phiên bản một hàm cùng chữ ký) |
| `v0_sequential.cpp` | `run_v0`: pipeline tuần tự 6 giai đoạn, đo thời gian từng giai đoạn |
| `common.h` | Kiểu dữ liệu dùng chung (`Image`, `Keypoint`, `Descriptor`, `Match`, `Homography`, …) và toàn bộ tham số `StitchConfig` |
| `stage1_preprocess.*` | Giai đoạn 1: grayscale BT.601 + Gaussian tách được |
| `stage2_keypoints.*` | Giai đoạn 2: Harris + NMS 3×3 + Grid Top-K |
| `stage3_orb.*` | Giai đoạn 3: ORB descriptor 256 bit |
| `stage4_matching.*` | Giai đoạn 4: Brute-force Hamming + Lowe ratio test + cross-check |
| `stage5_homography.*` | Giai đoạn 5: RANSAC + DLT 4 điểm + tinh chỉnh trên inlier |
| `stage6_warp_blend.*` | Giai đoạn 6: Inverse warping + bilinear + feathering |
| `image_io.h`, `stb_impl.cpp` | Đọc/ghi ảnh bằng `stb_image` / `stb_image_write` |
| `cuda_utils.cuh` | Macro kiểm tra lỗi CUDA, `GpuTimer`, in thông tin GPU (dùng cho V1+) |
| `stb_image.h`, `stb_image_write.h` | Thư viện stb (public domain, [nothings/stb](https://github.com/nothings/stb), commit `2c980bb`) |

Mỗi file `stageN_*.h` gồm các hàm `HOST_DEVICE` tính cho **một** pixel / keypoint / giả thuyết (dùng chung cho V0 và các kernel CUDA sau này), còn `stageN_*.cpp` là vòng lặp tuần tự của V0. Tham số của mọi giai đoạn nằm trong `StitchConfig` (`common.h`).

---

## 6. Build và chạy chương trình

### 6.1. Yêu cầu
- **Google Colab (khuyến nghị, môi trường chấm bài):** runtime GPU (T4), có sẵn CUDA Toolkit (`nvcc`) và `g++`.
- **Máy cá nhân (chỉ V0):** `g++` hỗ trợ C++17 (ví dụ MSYS2 UCRT64 trên Windows, hoặc g++ trên Linux). Không cần CUDA.
- Không dùng thư viện ngoài nào khác ngoài stb (đã có sẵn trong `src/`).

### 6.2. Build
```bash
# Colab / máy có CUDA (T4 -> sm_75); -O3 là bắt buộc vì nvcc không tự tối ưu code host
nvcc -O3 -std=c++17 -arch=sm_75 -o stitch src/*.cu src/*.cpp

# Máy không có CUDA: build bản CPU (chỉ chạy được --version 0)
g++ -O3 -std=c++17 -Isrc -x c++ src/main.cu -x none src/*.cpp -o stitch
```

### 6.3. Chạy
```bash
./stitch <left_image> <right_image> <output_image> --version <0|1|2|...> [--runs N]
```
- `<left_image>`: ảnh tham chiếu A; `<right_image>`: ảnh B (được biến dạng vào khung của A).
- `<output_image>`: định dạng theo đuôi file: `.png`, `.jpg` (quality 95) hoặc `.bmp`.
- `--runs N`: số lần đo (mặc định 5). Chương trình luôn chạy thêm **1 lần warm-up không tính giờ**, sau đó báo **trung bình và nhỏ nhất** của từng giai đoạn. Thời gian đọc/ghi ảnh không được tính.

Ví dụ:
```bash
./stitch "datasets_resized/medium/standard pairs/Outdoor/Pair 4/IMG_1129.png" \
         "datasets_resized/medium/standard pairs/Outdoor/Pair 4/IMG_1131.png" \
         output_outdoor4_medium.png --version 0 --runs 5
```

### 6.4. Kết quả in ra
Lần warm-up in thống kê của từng giai đoạn: số keypoint, số cặp ghép, số inlier RANSAC, ma trận H (B → A) và kích thước panorama. Sau các lần đo, chương trình in bảng thời gian:
```
Stage                          avg (ms)     min (ms)
----------------------------------------------------
1. Preprocessing                 ...          ...
2. Keypoint detection            ...          ...
3. Description                   ...          ...
4. Matching                      ...          ...
5. Homography (RANSAC)           ...          ...
6. Warping & blending            ...          ...
----------------------------------------------------
Total (end-to-end)               ...          ...
```
Chương trình trả về mã lỗi khác 0 nếu không đọc được ảnh, phiên bản chưa được cài đặt, hoặc không ước lượng được homography (ví dụ cặp ảnh quá ít đặc trưng chung).

### 6.5. Lưu ý
- **Windows + đường dẫn tiếng Việt có dấu:** bản build MinGW nhận tham số dòng lệnh theo bảng mã ANSI nên đường dẫn tuyệt đối chứa ký tự như `Năm` sẽ không mở được. Hãy chạy từ thư mục dự án với **đường dẫn tương đối** (như ví dụ trên). Colab/Linux không bị ảnh hưởng.
- Ảnh panorama có viền đen hình thang là bình thường: khi máy xoay lớn giữa hai lần chụp, ảnh B bị kéo giãn khi chiếu lên mặt phẳng của ảnh A.
