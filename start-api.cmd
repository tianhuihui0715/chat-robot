@echo off
setlocal

cd /d "%~dp0"

set "WSL_PROJECT_DIR=/mnt/e/project/chat-robot"

for /f %%I in ('powershell -NoProfile -Command "try { $resp = Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/health' -TimeoutSec 2; if ($resp.status -eq 'ok') { Write-Output RUNNING } else { Write-Output DOWN } } catch { Write-Output DOWN }"') do set "API_HEALTH=%%I"
if /I "%API_HEALTH%"=="RUNNING" (
  echo API service is already running on http://127.0.0.1:8000
  exit /b 0
)

echo Starting WSL API service on http://127.0.0.1:8000
wsl.exe --cd %WSL_PROJECT_DIR% python3 -c "import fastapi,uvicorn,pydantic,httpx,sqlalchemy,langsmith; import psycopg,qdrant_client,minio; from app.main import app" >nul 2>&1
if errorlevel 1 (
  echo Installing lightweight API packages in WSL...
  wsl.exe --cd %WSL_PROJECT_DIR% python3 -m pip install --upgrade pip
  if errorlevel 1 goto :fail
  wsl.exe --cd %WSL_PROJECT_DIR% python3 -m pip install fastapi httpx "uvicorn[standard]" pydantic python-dotenv langsmith sqlalchemy "psycopg[binary]" qdrant-client minio tomli
  if errorlevel 1 goto :fail
)

wsl.exe --cd %WSL_PROJECT_DIR% env RUNTIME_MODE=remote_inference INFERENCE_SERVICE_URL=http://127.0.0.1:8001 INFERENCE_TIMEOUT_SECONDS=120 TRACE_STORE_DSN=sqlite:///./data/chat_robot.db POSTGRES_DSN=postgresql+psycopg://chat_robot:chat_robot@127.0.0.1:5432/chat_robot QDRANT_URL=http://127.0.0.1:6333 MINIO_ENDPOINT=127.0.0.1:9000 MINIO_ACCESS_KEY=minioadmin MINIO_SECRET_KEY=minioadmin MINIO_SECURE=false MINIO_BUCKET=chat-robot EMBEDDING_MODEL_PATH= LANGSMITH_ENABLED=false python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo [ERROR] API service failed to start in WSL.
pause
exit /b 1
