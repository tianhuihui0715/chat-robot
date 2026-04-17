@echo off
setlocal

cd /d "%~dp0"

set "RUNTIME_MODE=remote_inference"
set "INFERENCE_SERVICE_URL=http://127.0.0.1:8001"
set "INFERENCE_TIMEOUT_SECONDS=120"

echo Starting Windows API service on http://127.0.0.1:8000
"D:\ProgramData\Anaconda3\envs\chat-robot-dev\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

