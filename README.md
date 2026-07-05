# FPT Smart Parking CV 🚗🏍️

Hệ thống thị giác máy tính (Computer Vision) dùng để đếm, nhận diện và định vị chỗ đậu xe tự động cho dự án **FPT Smart Parking**. 
Hệ thống ứng dụng trí tuệ nhân tạo (AI) kết hợp cùng kỹ thuật chiếu phối cảnh (BEV - Bird's Eye View) để giải quyết các thách thức trong thực tế của bãi đỗ xe như: phương tiện bị che khuất, phương tiện đỗ sai quy định và nhiễu tín hiệu từ các vật thể tĩnh (cây xanh, cột điện).

## 🌟 Các Tính Năng Nổi Bật (Features)
- **Bird's Eye View (BEV) Projection**: Chuyển đổi góc nhìn từ camera giám sát (góc xiên) sang hệ tọa độ 2D từ trên xuống, hỗ trợ việc tính toán khoảng cách và diện tích vị trí đỗ chính xác.
- **Smart Gap Detection (Nhận diện khoảng trống tự động)**: Thuật toán nhận diện phương tiện và tự động tính toán các khoảng trống (gaps) trên mặt phẳng BEV, sau đó ánh xạ trở lại hệ tọa độ khung hình của camera.
- **Exclusion Zones (Vùng loại trừ tĩnh)**: Cho phép thiết lập các khu vực loại trừ (như bồn cây, cột sắt, trụ điện). Các khoảng trống phát sinh trong những khu vực này sẽ bị hệ thống tự động loại bỏ.
- **Temporal Smoothing (Bộ lọc nhiễu thời gian)**: Đảm bảo một khoảng trống phải tồn tại liên tục qua một số lượng khung hình nhất định (ví dụ 15 frames) mới được ghi nhận, nhằm loại bỏ hoàn toàn hiện tượng nhiễu tín hiệu (flickering).
- **YOLOv8 X-Large (Độ chính xác cao)**: Tích hợp mô hình YOLOv8x từ Ultralytics, giúp tối ưu hóa việc phát hiện phương tiện ở độ phân giải 1280.
- **Real-time Metrics & Logging**: Ghi nhận tự động số lượng phương tiện (MC) và số khoảng trống (Gaps) tại từng khung hình ra tệp CSV, phục vụ cho việc thống kê và phân tích.

---

## 📁 Cấu Trúc Thư Mục (Directory Structure)
```
fpt-smart-parking-cv/
│
├── backend/
│   ├── core/
│   │   ├── detector.py
│   │   ├── bev_projector.py
│   │   ├── gap_detector.py
│   │   └── video_processor.py
│   └── config.yaml
│
├── data/
│   ├── raw_videos/
│   ├── calibration/
│   └── outputs/
│
├── models/
│
├── scripts/
│   ├── run_inference.py
│   ├── run_calibration.py
│   └── evaluate_metrics.py
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Yêu cầu Hệ thống (Prerequisites)
- Hệ điều hành: Windows 10/11, macOS, hoặc Linux.
- Python: Phiên bản **3.9** trở lên (Khuyến nghị 3.10 hoặc 3.11).
- Trình soạn thảo: VS Code, PyCharm.
- Yêu cầu phần cứng: Khuyến nghị sử dụng GPU hỗ trợ CUDA (NVIDIA RTX 3060 trở lên) để tối ưu hiệu suất, hoặc có thể vận hành trên CPU đối với các quá trình xử lý ngoại tuyến (Offline Inference).

---

## 🚀 Hướng Dẫn Cài Đặt (Installation)

**Bước 1: Tải mã nguồn về máy cục bộ**
```bash
git clone <đường-dẫn-repo-của-bạn>
cd fpt-smart-parking-cv
```

**Bước 2: Tạo môi trường ảo (Virtual Environment)**
```bash
python -m venv venv
venv\Scripts\activate
```
*(Đối với macOS/Linux)*:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện phụ thuộc**
```bash
pip install -r requirements.txt
```

---

## 📝 Thiết Lập Cấu Hình (Configuration)
Tệp cấu hình chính của hệ thống được đặt tại `backend/config.yaml`. Một số tham số quan trọng bao gồm:
- `pretrained_model`: Mặc định là `yolov8x.pt`. Có thể thay đổi thành `yolov8n.pt` để ưu tiên tốc độ xử lý.
- `inference_imgsz`: Độ phân giải đầu vào cho mô hình (mặc định 1280).
- `max_frames`: Giới hạn số lượng khung hình cần xử lý.
- `show_cv2`: Thiết lập thành `true` để hiển thị cửa sổ luồng video trong quá trình xử lý.
- `min_gap_frames`: Số lượng khung hình liên tục yêu cầu để hệ thống xác nhận một khoảng trống hợp lệ.

---

## ▶️ Hướng Dẫn Sử Dụng (Usage)

### 1. Vận Hành Luồng Xử Lý Chính (Main Inference Pipeline)

```bash
python scripts/run_inference.py
```

**Đầu ra hệ thống:**
1. Video minh họa kết quả: `data/outputs/annotated_video.mp4`
2. Tệp thống kê dữ liệu theo thời gian: `data/outputs/metrics.csv`
3. Tệp nhật ký chi tiết các khoảng trống: `data/outputs/gap_log.csv`

### 2. Thiết Lập Không Gian Bãi Đỗ (Calibration Tool)

```bash
python scripts/run_calibration.py
```
*Hướng dẫn:*
- Nhấn chuột trái để đánh dấu các điểm tọa độ.
- Nhấn phím `N` để chuyển sang bước tiếp theo.
- Nhấn phím `C` để xóa các điểm đã chọn và thực hiện lại.

### 3. Phân Tích Dữ Liệu (Analytics)

```bash
python scripts/evaluate_metrics.py
```
Biểu đồ phân tích sẽ được kết xuất tại: `data/outputs/parking_metrics_plot.png`.

---

## 💡 Xử Lý Lỗi Phổ Biến (Troubleshooting)
1. **Lỗi `ModuleNotFoundError`**: Kiểm tra lại quá trình cài đặt thư viện (`pip install -r requirements.txt`) và đảm bảo môi trường ảo đã được kích hoạt.
2. **Nhiễu nhận diện khoảng trống tại khu vực tĩnh**: Cập nhật `backend/config.yaml`, tăng giá trị `min_gap_frames` lên `15` hoặc `30` và sử dụng Calibration Tool để định nghĩa lại Vùng loại trừ (Exclusion Zone) tại vị trí đó.
3. **Hiệu năng xử lý thấp**: Điều chỉnh `config.yaml`, thay đổi `pretrained_model` thành `yolov8n.pt` và giảm `inference_imgsz` xuống `640` để cải thiện tốc độ xử lý.
