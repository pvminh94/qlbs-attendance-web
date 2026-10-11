#!/usr/bin/env bash
# ==============================================================================
# QLBS Biometric AI Engine — Bộ cài đặt tự động 1 dòng lệnh cho VPS 2 (AI Worker)
# Cổng mặc định: 8001
# ==============================================================================
set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}ℹ [INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}✔ [OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}⚠ [WARN]${NC} $1"; }
log_error() { echo -e "${RED}✖ [ERROR]${NC} $1" >&2; }

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}   QLBS BIOMETRIC AI ENGINE — TRÌNH CÀI ĐẶT MÁY CHỦ AI WORKER (VPS 2)  ${NC}"
echo -e "${BLUE}========================================================================${NC}"

if [[ "${EUID}" -ne 0 ]]; then
    log_error "Vui lòng chạy script với quyền root hoặc sudo: sudo bash deploy/install.sh"
    exit 1
fi

PORT=8001
INSTALL_DIR="/opt/qlbs-ai-engine"
REPO_URL="https://github.com/pvminh94/qlbs-attendance-web.git"

# 1. Kiểm tra Docker
if ! command -v docker &> /dev/null; then
    log_info "Đang cài đặt Docker Engine chính thức..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
    log_ok "Cài đặt Docker thành công."
fi

# 2. Chuẩn bị thư mục mã nguồn
if [[ ! -f "Dockerfile" ]]; then
    log_info "Đang tải mã nguồn AI Engine từ GitHub..."
    mkdir -p "${INSTALL_DIR}"
    if [[ -d "${INSTALL_DIR}/.git" ]]; then
        cd "${INSTALL_DIR}" && git pull origin main
    else
        git clone "${REPO_URL}" "${INSTALL_DIR}"
        cd "${INSTALL_DIR}"
    fi
fi

# 3. Tạo file .env
if [[ ! -f ".env" ]]; then
    echo "PORT=${PORT}" > .env
    echo "WORKERS=2" >> .env
    log_ok "Đã tạo cấu hình .env (Cổng: ${PORT})"
fi

# 4. Khởi chạy Docker Compose
log_info "Đang build và khởi động Container AI Engine (FastAPI + ONNX)..."
docker compose down --remove-orphans >/dev/null 2>&1 || true
docker compose up -d --build

# 5. Kiểm tra sức khỏe dịch vụ
log_info "Đang kiểm tra trạng thái dịch vụ tại http://localhost:${PORT}/health..."
ONLINE=false
for i in {1..30}; do
    if curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ONLINE=true
        break
    fi
    sleep 2
done

if [[ "${ONLINE}" == "true" ]]; then
    SERVER_IP=$(hostname -I | awk '{print $1}')
    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${GREEN}   ✔ DỊCH VỤ AI ENGINE (VPS 2) ĐÃ KHỞI CHẠY THÀNH CÔNG!                ${NC}"
    echo -e "${GREEN}========================================================================${NC}"
    echo -e " Địa chỉ Endpoint AI: ${CYAN}http://${SERVER_IP}:${PORT}${NC}"
    echo -e " Kiểm tra sức khỏe:   ${CYAN}http://${SERVER_IP}:${PORT}/health${NC}"
    echo -e ""
    echo -e " Cấu hình trên máy chủ chính QLBV (VPS 1):"
    echo -e " Thêm dòng sau vào file ${YELLOW}.env${NC} của QLBV:"
    echo -e " ${GREEN}AI_SERVICE_URL=http://${SERVER_IP}:${PORT}${NC}"
    echo -e "${GREEN}========================================================================${NC}"
else
    log_error "Dịch vụ AI chưa phản hồi sau 60 giây. Vui lòng kiểm tra log: docker compose logs"
    exit 1
fi
