# FPT Smart Parking CV - Git Workflow & SOP



## Cấu trúc thư mục dự án
* `/data/` - Chứa file ảnh/video test (Không push data quá lớn lên Git).
* `/models/` - Chứa file weights của YOLO (`.pt`, `.onnx`).
* `/scripts/` - Các script lẻ (extract_frames.py, get_coordinates.py).
* `/backend/` - Code FastAPI, WebSockets.
* `/frontend/` - Code ReactJS, UI Dashboard.
* `/notebooks/` - Nơi test nháp thuật toán bằng Jupyter Notebook.

## Quy tắc rẽ nhánh 
Mỗi task mới, tự tạo một nhánh từ nhánh `dev` với cú pháp: `<loai>/<ten-task>`
* Các loại nhánh: `feature` (tính năng mới), `bugfix` (sửa lỗi), `docs` (viết tài liệu).
* Ví dụ:
  * Train YOLO: `feature/yolo-baseline`
  * Làm BEV: `feature/bev-transform`
  * Làm API: `feature/fastapi-websocket`

## Quy trình code hàng ngày 

### Bước 1: Trước khi bắt đầu code
Luôn kéo code mới nhất từ nhánh `dev` về trước khi rẽ nhánh mới để tránh conflict.
git checkout dev
git pull origin dev
git checkout -b feature/ten-task

### Bước 2: Trong lúc code
Nhớ activate môi trường ảo. Nếu cài thêm thư viện gì mới (ví dụ `pip install ultralytics`), bắt buộc phải cập nhật lại file requirements:
pip freeze > requirements.txt

### Bước 3: Đẩy code lên (Commit & Push)
Quy tắc viết commit message: `[Loại] Mô tả ngắn gọn`.
Ví dụ: `[Feat] Thêm logic tính khoảng cách Euclidean` hoặc `[Fix] Sửa lỗi nhấp nháy ByteTrack`.
git add .
git commit -m "[Feat] Mô tả công việc đã làm"
git push origin feature/ten-task

## Quy trình ghép code (Pull Request - PR)
Khi code xong 1 tính năng và chạy test thành công trên máy cá nhân:
1. Lên GitHub, bấm nút **Compare & pull request**.
2. Chọn base branch là `dev`, compare branch là nhánh của bạn.
3. Có ít nhất 1 người Approve mới được bấm Merge vào nhánh `dev`.

## Quy tắc Clean Code 
* File `.gitignore` đã chặn thư mục `venv/`, `__pycache__/` và `node_modules/`. Tuyệt đối không cố tình force push các thư mục này lên git.
