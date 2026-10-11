#!/usr/bin/env bash
# ==============================================================================
# QLBS AI Engine — Kịch bản cập nhật tự động khi có code mới trên GitHub (VPS 2)
# ==============================================================================
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$(pwd)

echo "==> Đang kéo bản cập nhật AI Engine mới nhất từ GitHub..."
git config --global --add safe.directory "$ROOT" >/dev/null 2>&1 || true
git pull origin main

if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "✗ Không tìm thấy docker compose!" >&2
  exit 1
fi

echo "==> Đang build lại Container AI Engine..."
"${DC[@]}" up -d --build

echo "==> Dọn dẹp Docker images cũ..."
docker image prune -f >/dev/null 2>&1 || true

echo "==> Kiểm tra trạng thái AI Engine..."
sleep 2
curl -fs http://127.0.0.1:8001/health || true
echo ""
echo "✔ Cập nhật AI Engine thành công!"
