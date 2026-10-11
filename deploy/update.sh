#!/usr/bin/env bash
# ==============================================================================
# QLBS AI Engine — Kịch bản cập nhật tự động khi có code mới trên GitHub (VPS 2)
# ==============================================================================
set -Eeuo pipefail

echo "==> Đang kéo bản cập nhật AI Engine mới nhất từ GitHub..."
git pull origin main

echo "==> Đang build lại Container AI Engine..."
docker compose up -d --build

echo "==> Dọn dẹp Docker images cũ..."
docker image prune -f

echo "==> Kiểm tra trạng thái AI Engine..."
curl -fs http://127.0.0.1:8001/health || true
echo ""
echo "✔ Cập nhật AI Engine thành công!"
