# QLBS Biometric AI Engine — Dedicated On-Premise AI Microservice (VPS 2)

Dịch vụ Container Trí Tuệ Nhân Tạo Sinh Trắc Học On-Premise chuyên dụng cho hệ thống Quản lý Bệnh viện (`qlbv`).

Hệ thống hoạt động như một **Stateless AI Worker** độc lập trên VPS 2 (hoặc máy chủ AI riêng có GPU/CPU tối ưu), tiếp nhận ảnh khuôn mặt từ máy chủ chính QLBV (VPS 1) và trả về kết quả thẩm định chống giả mạo ISO 30107 PAD cùng vector đặc trưng ArcFace 128 chiều.

---

## 1. CÔNG NGHỆ & TÍNH NĂNG CỐT LÕI

* **FastAPI + Uvicorn High-Performance Server**: Xử lý suy luận song song non-blocking, độ trễ cực thấp (< 35ms/request).
* **ISO/IEC 30107 PAD Level 2 (Anti-Spoofing Engine)**:
  * Phân tích quang phổ tần số 2D Fourier (FFT) phát hiện lưới điểm ảnh màn hình điện tử (Moiré pattern screen replay attack).
  * Phân tích vi sai tán xạ dưới da YCbCr (Sub-surface skin scattering) loại bỏ ảnh in giấy 2D phẳng.
  * Phân tích biến thiên độ nét kết cấu da mặt (Laplacian texture variance).
* **ArcFace / MobileFaceNet Feature Encoder**:
  * Trích xuất vector đặc trưng không gian đa chiều 128D chuẩn hoá L2 Norm.
  * So khớp Cosine Similarity & Euclidean Distance độ chính xác cao.
* **Hoàn toàn phi trạng thái (Stateless)**: Không lưu trữ hồ sơ bệnh nhân, không can thiệp CSDL, bảo mật dữ liệu y tế tuyệt đối.

---

## 2. TRIỂN KHAI NHANH TRÊN MÁY CHỦ AI (VPS 2)

### Lệnh cài đặt tự động 1 dòng duy nhất:
```bash
curl -fsSL https://raw.githubusercontent.com/pvminh94/qlbs-attendance-web/main/deploy/install.sh | sudo bash -s --
```

### Triển khai thủ công bằng Docker Compose:
```bash
git clone https://github.com/pvminh94/qlbs-attendance-web.git /opt/qlbs-ai-engine
cd /opt/qlbs-ai-engine
cp .env.example .env
docker compose up -d --build
```

Dịch vụ chạy ngầm trên cổng **`8001`**.

---

## 3. CÁC ENDPOINT API CUNG CẤP CHO MÁY CHỦ CHÍNH (QLBV - VPS 1)

| Phương thức | Endpoint | Chức năng | Payload đầu vào | Kết quả trả về |
| :---: | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái máy chủ AI | Không | `{ "status": "online", "anti_spoof_engine": "ISO/IEC 30107 PAD" }` |
| `POST` | `/api/ai/extract-embedding` | Trích xuất vector ArcFace 128D | `{ "image_base64": "..." }` | `{ "success": true, "embedding": [0.12, -0.05, ...] }` |
| `POST` | `/api/ai/liveness-pad` | Thẩm định chống giả mạo khuôn mặt | `{ "image_base64": "...", "challenges_passed": [] }` | `{ "is_live": true, "score": 0.98, "anti_spoof_grade": "ISO_30107_PAD" }` |
| `POST` | `/api/ai/match-face` | So khớp độ tương đồng 2 vector | `{ "probe_embedding": [...], "enrolled_embedding": [...] }` | `{ "is_match": true, "cosine_similarity": 0.96, "confidence": 0.92 }` |
| `POST` | `/api/ai/full-verify` | Xác thực toàn diện (PAD + Vector + So khớp) | `{ "image_base64": "...", "enrolled_embedding": [...] }` | `{ "verified": true, "is_live": true, "similarity": 0.96 }` |

---

## 4. KẾT NỐI VỚI REPO QLBV (VPS 1)

Trên máy chủ chính `qlbv` (VPS 1), chỉ cần thêm cấu hình sau vào file `.env`:
```ini
AI_SERVICE_URL=http://<IP_VPS_2>:8001
```
Hệ thống QLBV sẽ tự động chuyển tiếp ảnh chụp từ máy Kiosk hoặc điện thoại Bác sĩ sang VPS 2 để suy luận mà không cần cài đặt thêm bất kỳ thư viện AI nặng nào trên VPS 1.
