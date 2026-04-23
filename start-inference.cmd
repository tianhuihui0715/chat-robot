@echo off
setlocal

cd /d "%~dp0"

echo Starting WSL inference service on http://127.0.0.1:8001
wsl.exe bash -lc "cd /mnt/e/project/chat-robot && export INFERENCE_RUNTIME_MODE=local_hf && export HOST_MODEL_ROOT=/root/models && export LLM_MODEL_PATH=/root/models/Qwen3-8B && export INTENT_MODEL_PATH=/root/models/Qwen2.5-1.5B-Instruct && uvicorn app.inference.main:app --host 0.0.0.0 --port 8001 --no-access-log"
