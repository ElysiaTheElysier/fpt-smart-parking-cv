# FPT Smart Parking CV

Hệ thống giám sát trạng thái đỗ xe máy bằng Deep Learning.

## Giới thiệu

Đây là prototype cho bài toán **giám sát bãi đỗ xe máy không có ô cố định** — một bài toán phổ biến tại Việt Nam, nơi xe máy thường đỗ tự do sát nhau trong bãi xe trường học, tòa nhà, siêu thị.

### Bài toán

- Xe máy đỗ không theo ô cố định, sát nhau, dễ bị che khuất (occlusion).
- Cần đếm xe, phát hiện khoảng trống còn đỗ được, và nhận diện xe đỗ sai vị trí.
- Camera cố định, góc chéo từ trên xuống.

### Dữ liệu

- **1 video điện thoại** duy nhất, quay trong bãi xe thực tế.
- Thời lượng: **4 phút 33 giây**, **60 FPS**.
- Nội dung: xe máy đang đỗ, người đi qua, người lấy xe.

## Phương pháp

| Thành phần | Mô tả |
|------------|-------|
| **YOLO (YOLOv8)** | Phát hiện xe máy và người. Dùng pretrained COCO, có thể fine-tune trên dữ liệu Roboflow. |
| **Bird's Eye View (BEV)** | Perspective transform để giảm méo phối cảnh, đo khoảng cách đồng đều hơn. |
| **Tracking (ByteTrack)** | Giữ track ID cho mỗi xe, giảm nhấp nháy khi người đi qua hoặc xe bị che tạm thời. |
| **Gap Measurement** | Đo khoảng trống giữa các xe dựa trên bottom-center trong BEV space. |
| **Parking Rules** | Rule đơn giản: xe nằm ngoài vùng cho phép → cảnh báo. |

## Cấu trúc thư mục

```
fpt-smart-parking-cv/
├── backend/
│   ├── config.yaml              # Config toàn pipeline
│   └── core/
│       ├── bev.py               # Perspective transform
│       ├── detector.py          # YOLO wrapper
│       ├── tracker.py           # IoU tracker fallback
│       ├── gap_measurement.py   # Đo khoảng trống
│       ├── parking_rules.py     # Luật đỗ xe
│       ├── video_processor.py   # Pipeline chính
│       └── utils.py             # Tiện ích dùng chung
├── frontend/
│   └── streamlit_app.py         # Dashboard demo
├── data/
│   ├── raw_videos/              # Video gốc
│   ├── extracted_frames/        # Frame đã extract
│   ├── calibration/             # BEV + ROI points
│   └── outputs/                 # Video annotated, CSV, screenshots
├── models/                      # YOLO weights
├── scripts/
│   ├── extract_frames.py        # Extract frame từ video
│   ├── export_roboflow_frames.py # Export ảnh cho Roboflow
│   ├── select_bev_points.py     # Chọn 4 điểm BEV
│   ├── select_roi.py            # Chọn ROI polygon
│   ├── run_inference.py         # Chạy pipeline
│   ├── evaluate.py              # Đánh giá kết quả
│   └── train_yolo.py            # Train YOLO custom
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
# Clone repo
git clone https://github.com/ElysiaTheElysier/fpt-smart-parking-cv.git
cd fpt-smart-parking-cv

# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Cài thư viện
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 1. Đặt video đầu vào

Đặt file video vào:
```
data/raw_videos/input_video.mp4
```

### 2. Extract frame để kiểm tra

```bash
python scripts/extract_frames.py --video data/raw_videos/input_video.mp4 --fps 1 --output data/extracted_frames
```

### 3. Export ảnh để label trên Roboflow (tùy chọn)

```bash
python scripts/export_roboflow_frames.py --video data/raw_videos/input_video.mp4 --fps 1 --max-frames 120 --output data/roboflow_upload
```

Sau đó upload lên [Roboflow](https://app.roboflow.com), label class `motorcycle` và `person`, export format YOLOv8.

### 4. Chọn BEV points

```bash
python scripts/select_bev_points.py --video data/raw_videos/input_video.mp4 --output data/calibration/bev_points.json
```

Click 4 điểm mặt đất theo thứ tự: Top-Left → Top-Right → Bottom-Right → Bottom-Left.
Phím: `r` = reset, `s` = save, `q` = quit.

### 5. Chọn ROI (vùng bãi xe)

```bash
python scripts/select_roi.py --video data/raw_videos/input_video.mp4 --output data/calibration/roi_points.json
```

Click quanh vùng bãi xe cần phân tích. Tối thiểu 3 điểm.

### 6. Chạy inference

```bash
python scripts/run_inference.py --config backend/config.yaml
```

Kết quả:
- `data/outputs/annotated_video.mp4` — video đã annotate
- `data/outputs/metrics.csv` — thống kê theo frame
- `data/outputs/gap_log.csv` — log khoảng trống

### 7. Đánh giá kết quả

```bash
python scripts/evaluate.py --metrics data/outputs/metrics.csv --gaps data/outputs/gap_log.csv --plot
```

### 8. Mở dashboard

```bash
streamlit run frontend/streamlit_app.py
```

### 9. Train YOLO custom (tùy chọn)

```bash
python scripts/train_yolo.py --data data/yolo_dataset/data.yaml --epochs 50 --imgsz 640
```

## Hạn chế

- Chỉ có **một góc camera** duy nhất — không thể xử lý toàn bộ bãi xe lớn.
- BEV hiện dùng **pixel threshold**, chưa có calibration chính xác mét/cm.
- Điểm tiếp đất (bottom-center của bbox) chỉ là **xấp xỉ** — không phải vị trí bánh xe thật.
- **Occlusion nặng** (xe bị che >70%) vẫn có thể bị miss detection.
- Chưa phân biệt xe đang đỗ vs. xe đang di chuyển.
- Tracking có thể bị gán sai ID khi xe bị che rồi xuất hiện lại.

## Hướng phát triển

- Thêm nhiều video từ nhiều góc camera.
- Train YOLO custom trên dữ liệu Roboflow để tăng accuracy.
- Dùng **instance segmentation** thay vì bounding box để xác định vị trí chính xác hơn.
- Calibration bằng **vật chuẩn thật** (giấy A4, thước) để đo khoảng cách thực tế.
- Export model sang **ONNX** để deploy nhẹ hơn.
- Tích hợp **Grad-CAM / XAI** để giải thích prediction (optional).
- Thêm cảnh báo realtime qua WebSocket.

## Tác giả

Đồ án sinh viên — FPT University.
