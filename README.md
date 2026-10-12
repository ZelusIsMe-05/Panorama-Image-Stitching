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

Công cụ tiền xử lý yêu cầu Python 3 và thư viện `Pillow`:

```bash
pip install -r tools/requirements.txt
```

---

## 3. Hướng dẫn sử dụng công cụ resize (`tools/resize_images.py`)

Công cụ hỗ trợ các định dạng: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff` (loại bỏ các file lạ như `.pnm`).

### 3.1. Resize 1 ảnh về 1 kích thước cụ thể
```bash
# Về mức small (720p)
python tools/resize_images.py -i duong_dan_anh.jpg -o dataset/small -r small

# Về mức medium (1080p)
python tools/resize_images.py -i duong_dan_anh.jpg -o dataset/medium -r medium

# Về mức large (4K)
python tools/resize_images.py -i duong_dan_anh.jpg -o dataset/large -r large
```

### 3.2. Resize 1 ảnh ra cả 3 kích thước cùng lúc
```bash
python tools/resize_images.py -i duong_dan_anh.jpg -o dataset -r all
```
Ảnh xuất ra sẽ tự động được chia vào các thư mục:
- `dataset/small/duong_dan_anh.jpg`
- `dataset/medium/duong_dan_anh.jpg`
- `dataset/large/duong_dan_anh.jpg`

### 3.3. Xử lý hàng loạt toàn bộ thư mục ảnh
```bash
python tools/resize_images.py -i raw_photos -o dataset -r all
```
Lệnh trên sẽ quét toàn bộ ảnh hợp lệ trong thư mục `raw_photos/` và xuất cả 3 kích thước vào `dataset/`.

### 3.4. Các tùy chọn khác
- **Độ phân giải tự chọn:** `-r 800x600`
- **Không crop 16:9 (chỉ resize trực tiếp):** thêm `--no-crop`
