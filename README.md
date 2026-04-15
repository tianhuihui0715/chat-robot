# Chat Robot

本仓库用于搭建一个本地智能聊天系统，目标架构是：

- `Qwen3-8B` 作为基座模型
- `Qwen2.5-1.5B-Instruct` 作为意图识别/路由模型
- `bge-m3` 作为 embedding 模型
- `bge-reranker-v2-m3` 作为重排模型
- `FastAPI` 作为服务入口
- 单并发 GPU 生成队列，避免 12GB 显存被并发请求压爆

## 当前状态

当前仓库提供的是第一版可运行骨架：

- `POST /api/v1/chat`：对话接口
- `POST /api/v1/knowledge/ingest`：导入知识文档
- `GET /api/v1/health`：健康检查
- 内置 mock 意图识别、mock 检索和单并发生成队列

这版重点是先把服务形态、目录结构和调用链路搭起来，后续再逐步替换成真实模型加载逻辑。

## 目录结构

```text
app/
  api/                FastAPI 路由
  core/               配置和日志
  schemas/            请求/响应模型
  services/           意图识别、检索、生成、队列、流程编排
config/               静态默认配置（TOML）
pyproject.toml        项目元数据、依赖、工具链配置
requirements.txt      兼容入口，内部转发到 pyproject
requirements-train.txt 训练依赖兼容入口
.env.example          本地配置示例
```

## 快速启动

1. 安装依赖

```bash
pip3 install -e .
```

2. 复制配置

```bash
cp .env.example .env
```

3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4. 打开文档

```text
http://127.0.0.1:8000/docs
```

## 配置分层

当前项目使用三层配置：

- `pyproject.toml`
  用来维护项目版本、依赖和工具链配置
- `config/*.toml`
  用来维护静态默认配置，例如模型路径默认值、RAG 参数、数据库默认地址
- `.env`
  用来覆盖运行时配置，例如本机模型目录、Docker 地址、密码和运行模式

推荐原则：

- 会随环境变化的内容放 `.env`
- 项目级默认策略放 `config/*.toml`
- 依赖和版本放 `pyproject.toml`

## Docker 部署

这套项目适合用 `docker compose` 跑成多服务：

- `api`：FastAPI 应用
- `postgres`：业务数据、会话、日志
- `qdrant`：向量库

模型不打进镜像，而是从宿主机目录挂载到容器内。对你当前的 WSL 环境，默认挂载目录是 `/root/models`。

1. 准备配置

```bash
cp .env.example .env
```

2. 构建并启动

```bash
docker compose up --build -d
```

3. 查看状态

```bash
docker compose ps
docker compose logs -f api
```

4. 停止服务

```bash
docker compose down
```

如果你要保留数据库和向量库数据，不要加 `-v`。

## 后续开发顺序

1. 将 mock 生成服务替换为 `Qwen3-8B 4bit` 实际加载
2. 将 mock 意图识别替换为 `Qwen2.5-1.5B-Instruct`
3. 将内存检索替换为 `bge-m3 + FAISS`
4. 增加 `bge-reranker-v2-m3` 重排
5. 增加会话存储、日志和评测脚本
