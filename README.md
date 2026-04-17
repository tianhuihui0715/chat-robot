# Chat Robot

一个面向本地部署场景的智能聊天系统骨架，当前采用“方案 B”：

- `api` 主服务负责业务编排、请求 trace、本地流程复现和后续 RAG/工具调用
- `inference` 独立推理服务负责模型加载与生成，便于把 WSL/GPU 环境和业务服务解耦
- `postgres` 负责业务数据
- `qdrant` 负责向量检索

## 当前状态

目前仓库已经具备这些能力：

- `POST /api/v1/chat`
- `POST /api/v1/knowledge/ingest`
- `GET /api/v1/health`
- 本地 trace 骨架持久化，可脱离 LangSmith 复现流程
- `api -> inference` 远程生成链路
- 独立推理服务入口：`app.inference.main:app`

当前仍然保留 `mock` 运行模式，便于在没有模型环境的机器上继续开发主服务。

## 架构说明

### API 主服务

主服务负责：

- 接收对话请求
- 记录 request trace / trace steps / 各节点明细
- 后续承载意图识别、RAG、工具调用
- 通过 HTTP 调用独立推理服务

### Inference 推理服务

推理服务负责：

- 独立加载 `Qwen3-8B`
- 接收标准化生成请求
- 返回最终回答
- 与业务服务解耦，便于单独部署到 WSL/GPU 环境

当前已支持两种推理服务模式：

- `mock`
- `local_hf`

## 观测设计

项目采用双轨观测：

- `LangSmith`
  - 仅记录轻量链路轨迹和映射关系
- 本地数据库
  - 记录 request trace 骨架
  - 记录步骤骨架
  - 记录意图、检索、生成等节点明细

这样即使脱离 LangSmith 云平台，也可以在后台复现一次请求流程。

## 配置分层

- `pyproject.toml`
  - 项目元数据、依赖、工具链
- `config/*.toml`
  - 静态默认配置
  - 其中 `config/intents.toml` 用于管理意图识别提示词和 few-shot 示例，方便直接调参对比
- `.env`
  - 当前环境覆盖项

推荐原则：

- 模型路径、生成参数、数据库地址这类运行参数优先放 `config/*.toml` 和 `.env`
- `docker-compose.yml` 只描述部署拓扑、容器依赖、卷挂载和少量运行模式覆盖

## 关键配置

### API 主服务

- `RUNTIME_MODE`
  - `mock`
  - `remote_inference`

### 推理服务

- `INFERENCE_RUNTIME_MODE`
  - `mock`
  - `local_hf`

### 服务间调用

- `INFERENCE_SERVICE_URL`
- `INFERENCE_TIMEOUT_SECONDS`

## 本地启动

安装依赖：

```bash
pip3 install -e .
```

复制配置：

```bash
cp .env.example .env
```

启动 API 主服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动独立推理服务：

```bash
uvicorn app.inference.main:app --host 0.0.0.0 --port 8001 --reload
```

## Docker Compose

当前 `docker-compose.yml` 已按方案 B 拆分：

- `api`
- `inference`
- `postgres`
- `qdrant`

其中：

- 模型路径和生成参数默认读取 `config/models.toml` 或 `.env`
- `compose` 只额外覆盖
  - `RUNTIME_MODE=remote_inference`
  - `INFERENCE_SERVICE_URL=http://inference:8001`
  - `INFERENCE_RUNTIME_MODE=local_hf`

启动：

```bash
docker compose up --build -d
```

查看状态：

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f inference
```

## Windows API only setup

When the model service is already running in WSL, the Windows host can use a
lightweight Python environment for the API service only.

```powershell
py -3.10 -m venv .venv-api
.venv-api\Scripts\python -m pip install --upgrade pip
.venv-api\Scripts\python -m pip install -r requirements-api.txt
```

Start the API service from the repository root:

```powershell
$env:RUNTIME_MODE="remote_inference"
$env:INFERENCE_SERVICE_URL="http://127.0.0.1:8001"
$env:INFERENCE_TIMEOUT_SECONDS="120"
.venv-api\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 后续开发顺序

1. 将 `inference` 服务的 `local_hf` 跑通并稳定加载 `Qwen3-8B 4bit`
2. 将意图识别模型也迁入独立推理服务或独立路由服务
3. 将主服务检索链路替换为 `bge-m3 + Qdrant`
4. 接入 `bge-reranker-v2-m3`
5. 增加后台 trace 查询接口和页面
6. 再继续推进工具系统、双检索、评测与蒸馏
