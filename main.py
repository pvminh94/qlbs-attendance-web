"""
QLBS Biometric AI Engine — Dịch vụ Container Trí Tuệ Nhân Tạo Sinh Trắc Học On-Premise
FastAPI Server chạy trên cổng 8001 (CPU/GPU-ready qua ONNX Runtime / NumPy)
"""
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.anti_spoof import anti_spoof_engine
from models.face_encoder import face_encoder

app = FastAPI(
    title="QLBS On-Premise Biometric AI Engine",
    description="Dịch vụ suy luận mô hình chống giả mạo khuôn mặt (ISO 30107 PAD) và trích xuất vector ArcFace nội bộ",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DTO Models ---
class LivenessRequest(BaseModel):
    image_base64: str = Field(..., description="Ảnh chân dung chụp từ camera điện thoại (base64)")
    challenges_passed: Optional[List[str]] = Field(default=[], description="Danh sách các thử thách động đã đạt")
    flash_reflection_delta: Optional[float] = Field(default=None, description="Độ biến thiên phản xạ quang học màn hình")

class ExtractEmbeddingRequest(BaseModel):
    image_base64: str = Field(..., description="Ảnh khuôn mặt để trích xuất vector 128D")

class MatchFaceRequest(BaseModel):
    probe_embedding: List[float] = Field(..., description="Vector khuôn mặt cần xác thực")
    enrolled_embedding: List[float] = Field(..., description="Vector khuôn mặt đã lưu trong CSDL máy chủ")

class FullVerifyRequest(BaseModel):
    image_base64: str
    enrolled_embedding: List[float]
    challenges_passed: Optional[List[str]] = []

# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "qlbs-biometric-ai-engine",
        "version": "1.2.0-on-premise",
        "anti_spoof_engine": "ISO/IEC 30107 PAD Level 2",
        "face_model": "ArcFace / MobileFaceNet 128D L2",
        "execution_provider": "CPU-Optimized (ONNX/NumPy)",
    }

@app.post("/api/ai/liveness-pad")
def check_liveness(req: LivenessRequest):
    """
    Kiểm tra người thật và chống giả mạo (Passive + Active Liveness PAD)
    Phát hiện ảnh in giấy 2D, màn hình phát lại Replay Attack, hoặc mặt nạ tĩnh
    """
    res = anti_spoof_engine.analyze_image_base64(req.image_base64)

    # Nếu có thử thách động gửi lên, bổ sung vào kết quả thẩm định
    if req.challenges_passed and len(req.challenges_passed) >= 2:
        if res["is_live"]:
            res["score"] = min(0.99, res["score"] + 0.05)

    return res

@app.post("/api/ai/extract-embedding")
def extract_embedding(req: ExtractEmbeddingRequest):
    """
    Trích xuất vector đặc trưng khuôn mặt 128 chiều chuẩn hoá L2 Norm
    """
    res = face_encoder.extract_embedding_from_base64(req.image_base64)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res.get("error", "Lỗi trích xuất vector"))
    return res

@app.post("/api/ai/match-face")
def match_face(req: MatchFaceRequest):
    """
    So khớp độ tương đồng giữa 2 vector khuôn mặt (Cosine Similarity & Euclidean Distance)
    """
    if len(req.probe_embedding) != 128 or len(req.enrolled_embedding) != 128:
        raise HTTPException(
            status_code=400,
            detail=f"Kích thước vector không hợp lệ (cần 128 chiều, nhận: {len(req.probe_embedding)} và {len(req.enrolled_embedding)})"
        )
    return face_encoder.compare_embeddings(req.probe_embedding, req.enrolled_embedding)

@app.post("/api/ai/full-verify")
def full_verify(req: FullVerifyRequest):
    """
    Xác thực toàn diện một chạm (< 50ms): Chống giả mạo PAD + Trích xuất vector + So khớp mẫu CSDL
    """
    t0 = time.time()

    # 1. Chống giả mạo PAD
    liveness_res = anti_spoof_engine.analyze_image_base64(req.image_base64)
    if not liveness_res["is_live"]:
        return {
            "verified": False,
            "is_live": False,
            "reason": liveness_res.get("reason", "Phát hiện giả mạo khuôn mặt"),
            "spoof_type": liveness_res.get("spoof_type"),
            "liveness_score": liveness_res["score"],
            "total_latency_ms": round((time.time() - t0) * 1000, 2),
        }

    # 2. Trích xuất vector từ ảnh chụp
    extract_res = face_encoder.extract_embedding_from_base64(req.image_base64)
    if not extract_res["success"]:
        return {
            "verified": False,
            "is_live": True,
            "reason": "Không thể trích xuất vector từ ảnh khuôn mặt",
            "total_latency_ms": round((time.time() - t0) * 1000, 2),
        }

    # 3. So khớp với vector mẫu đã đăng ký của Bác sĩ trong CSDL máy chủ
    match_res = face_encoder.compare_embeddings(extract_res["embedding"], req.enrolled_embedding)

    total_latency = round((time.time() - t0) * 1000, 2)

    return {
        "verified": match_res["is_match"],
        "is_live": True,
        "similarity": match_res["cosine_similarity"],
        "euclidean_distance": match_res["euclidean_distance"],
        "confidence": match_res["confidence"],
        "liveness_score": liveness_res["score"],
        "total_latency_ms": total_latency,
        "reason": "Xác thực bác sĩ chính chủ thành công" if match_res["is_match"] else "Khuôn mặt không khớp với hồ sơ bác sĩ",
    }

if __name__ == "__main__":
    import uvicorn
    print("[QLBS AI Engine] Khởi động dịch vụ AI On-Premise trên cổng 8001...")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
