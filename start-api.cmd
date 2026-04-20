@echo off
setlocal

cd /d "%~dp0"

set "RUNTIME_MODE=remote_inference"
set "INFERENCE_SERVICE_URL=http://127.0.0.1:8001"
set "INFERENCE_TIMEOUT_SECONDS=120"
set "POSTGRES_DSN=postgresql+psycopg://chat_robot:chat_robot@127.0.0.1:5432/chat_robot"
set "QDRANT_URL=http://127.0.0.1:6333"
set "MINIO_ENDPOINT=127.0.0.1:9000"
set "MINIO_ACCESS_KEY=minioadmin"
set "MINIO_SECRET_KEY=minioadmin"
set "MINIO_SECURE=false"
set "MINIO_BUCKET=chat-robot"

echo Starting Windows API service on http://127.0.0.1:8000
"D:\ProgramData\Anaconda3\envs\chat-robot-dev\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

