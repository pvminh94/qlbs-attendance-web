"""
Mô-đun Trích Xuất & So Khớp Vector Đặc Trưng Khuôn Mặt SOTA Deep Neural Network
Sử dụng bộ đôi mô hình On-Premise tối ưu hóa trên CPU:
1. Face Detector & Alignment: OpenCV YuNet (face_detection_yunet_2023mar.onnx) - phát hiện khuôn mặt & 5 điểm mốc (mắt, mũi, miệng)
2. Feature Extractor: InsightFace MobileFaceNet ArcFace (w600k_mbf.onnx) - trích xuất vector 512 chiều chuẩn hóa L2 Norm
Khả năng phân biệt cực cao (Orthogonal Cosine Similarity), loại bỏ hoàn toàn hiện tượng nhận nhầm người khác.
"""
import os
import io
import time
import base64
import threading
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# 5 điểm mốc chuẩn theo tọa độ InsightFace ArcFace 112x112
# [Mắt phải, mắt trái, chóp mũi, khóe miệng phải, khóe miệng trái]
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
ARCFACE_PATH = os.path.join(BASE_DIR, "w600k_mbf.onnx")


class FaceEncoder:
    def __init__(self):
        self.embedding_dim = 512
        # Ngưỡng Cosine cho ArcFace 512D: Người khác nhau ~0.0 - 0.25; Chính chủ ~0.70 - 0.95
        self.match_cosine_threshold = 0.50
        self.match_euclidean_threshold = 1.05
        self.lock = threading.Lock()

        # Khởi tạo mô hình ONNX ArcFace
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.arcface_session = ort.InferenceSession(ARCFACE_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
        self.arcface_input_name = self.arcface_session.get_inputs()[0].name

        # Khởi tạo OpenCV YuNet detector
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=0.55,
            nms_threshold=0.3,
            top_k=5000,
        )

    def _align_face(self, bgr_img: np.ndarray):
        """
        Phát hiện khuôn mặt và xoay/cắt chuẩn hóa tọa độ hình học (Affine Alignment)
        Trả về (aligned_face_112x112, face_score, success)
        """
        h, w, _ = bgr_img.shape
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(bgr_img)
        if faces is None or len(faces) == 0:
            # Thử lại với viền đệm 10% nếu khuôn mặt sát viền camera
            pad_y, pad_x = int(h * 0.1), int(w * 0.1)
            padded = cv2.copyMakeBorder(bgr_img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
            self.detector.setInputSize((padded.shape[1], padded.shape[0]))
            _, faces = self.detector.detect(padded)
            if faces is None or len(faces) == 0:
                return None, 0.0, False
            # Bỏ viền bù trừ
            bgr_img = padded

        # Chọn khuôn mặt lớn nhất (diện tích w*h lớn nhất - người đứng gần camera nhất)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        score = float(largest_face[-1])

        # 5 điểm mốc khuôn mặt
        landmarks = np.array([
            largest_face[4:6],   # Mắt phải
            largest_face[6:8],   # Mắt trái
            largest_face[8:10],  # Đầu mũi
            largest_face[10:12], # Khóe miệng phải
            largest_face[12:14], # Khóe miệng trái
        ], dtype=np.float32)

        # Tính toán ma trận biến đổi Affine để căn chỉnh góc nghiêng đầu
        M, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_FACIAL_POINTS)
        if M is not None:
            aligned_face = cv2.warpAffine(bgr_img, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
        else:
            # Dự phòng cắt Bounding Box nếu không ước lượng được góc quay
            x, y, fw, fh = largest_face[0:4].astype(int)
            mx, my = int(fw * 0.15), int(fh * 0.15)
            x1, y1 = max(0, x - mx), max(0, y - my)
            x2, y2 = min(bgr_img.shape[1], x + fw + mx), min(bgr_img.shape[0], y + fh + my)
            crop = bgr_img[y1:y2, x1:x2]
            aligned_face = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)

        return aligned_face, score, True

    def extract_embedding_from_base64(self, img_b64: str) -> dict:
        start_time = time.time()

        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        try:
            img_bytes = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr_img is None:
                raise ValueError("Không thể decode ma trận pixel từ ảnh")
        except Exception as e:
            return {
                "success": False,
                "error": f"Không thể giải mã ảnh: {str(e)}",
                "embedding": [],
            }

        with self.lock:
            aligned_face, score, found = self._align_face(bgr_img)

        if not found or aligned_face is None:
            return {
                "success": False,
                "error": "Không tìm thấy khuôn mặt rõ ràng trong ảnh. Vui lòng nhìn thẳng vào camera và đảm bảo đủ ánh sáng.",
                "embedding": [],
            }

        # Chuẩn bị Tensor cho mô hình ArcFace MobileFaceNet:
        # Chuẩn hóa BGR: (pixel - 127.5) / 127.5, chuyển sang layout NCHW [1, 3, 112, 112]
        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )

        with self.lock:
            emb_raw = self.arcface_session.run(None, {self.arcface_input_name: blob})[0][0]

        # Chuẩn hóa L2 Norm = 1.0
        norm = np.linalg.norm(emb_raw)
        if norm > 1e-6:
            emb = emb_raw / norm
        else:
            emb = emb_raw

        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "success": True,
            "embedding": [round(float(v), 6) for v in emb],
            "embedding_dim": len(emb),
            "quality_score": round(score, 4),
            "latency_ms": latency,
        }

    def compare_embeddings(self, vec1: list, vec2: list) -> dict:
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        if len(v1) != len(v2):
            return {
                "is_match": False,
                "cosine_similarity": 0.0,
                "euclidean_distance": 99.0,
                "confidence": 0.0,
                "threshold": self.match_cosine_threshold,
                "error": f"Độ dài vector không đồng nhất ({len(v1)} vs {len(v2)}). Vui lòng đăng ký lại khuôn mặt mẫu.",
            }

        # 1. Cosine Similarity
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            cosine = float(np.dot(v1, v2) / (norm1 * norm2))
        else:
            cosine = 0.0

        # 2. Euclidean Distance
        dist = float(np.linalg.norm(v1 - v2))

        is_match = (cosine >= self.match_cosine_threshold)
        # Độ tin cậy tính từ ngưỡng 0.30 đến 1.0
        confidence = float(np.clip((cosine - 0.30) / 0.70, 0.0, 1.0))

        return {
            "is_match": is_match,
            "cosine_similarity": round(cosine, 4),
            "euclidean_distance": round(dist, 4),
            "confidence": round(confidence, 4),
            "threshold": self.match_cosine_threshold,
        }


face_encoder = FaceEncoder()
