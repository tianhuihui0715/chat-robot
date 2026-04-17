@echo off
setlocal

echo Starting WSL inference service on http://127.0.0.1:8001
wsl.exe bash -lc "cd /mnt/e/project/chat-robot && uvicorn app.inference.main:app --host 0.0.0.0 --port 8001"

