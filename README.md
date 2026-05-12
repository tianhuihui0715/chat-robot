# 开发记录

## 2026-04-17

### 1. 收口意图识别基础路线
**开发内容**
- 明确意图识别不只判断“是什么问题”，还要决定后续走哪条执行路径
- 围绕以下模式逐步收口：
  - `direct`
  - `rag`
  - `plan_execute`
  - `clarify`
- 调整意图识别规则、提示词和少量回归逻辑

**遇到的问题**
- 早期链路更偏固定流程，复杂问题和追问问题边界不清楚
- 小模型容易把模糊问题、复杂问题、普通知识问答混在一起

**处理方式**
- 将意图识别从“分类”扩展为“执行路由决策”
- 明确 `clarify` 是对话控制分支，不是普通工具
- 明确复杂问题后续应往规划执行链发展，而不是继续堆普通 RAG

**涉及模块**
- `E:\project\chat-robot\app\services\intent_service.py`
- `E:\project\chat-robot\app\inference\backends.py`
- `E:\project\chat-robot\config\intents.toml`

---

### 2. 收口 follow-up / clarify 的处理方式
**开发内容**
- 讨论并明确了：
  - `follow_up` 作为延续上文问题的意图
  - `clarify` 作为信息不足时的追问动作
- 不把追问做成普通工具

**遇到的问题**
- 复杂问题和信息不足问题都可能被误送进普通检索链
- 如果把“追问”也做成工具，语义会很乱

**处理方式**
- 让 `clarify` 保留在对话控制层
- 让工具层只承载真实外部能力

**涉及模块**
- `E:\project\chat-robot\app\services\intent_service.py`
- `E:\project\chat-robot\app\schemas\chat.py`

---

## 2026-04-20

### 1. 推进基础 RAG 与聊天主链路
**开发内容**
- 推进聊天链路、知识库、RAG 基础能力
- 聊天界面、知识库接入和基础检索逐步成型

**遇到的问题**
- 需要在普通聊天、知识问答、知识库查询之间建立基础链路
- 系统还没有复杂任务执行能力，问题主要集中在基础可用性

**处理方式**
- 先把主链路搭起来，确保：
  - 能问
  - 能检索
  - 能看到基础结果

**涉及模块**
- `E:\project\chat-robot\app\services\chat_pipeline.py`
- `E:\project\chat-robot\app\services\retriever_service.py`
- `E:\project\chat-robot\app\static\chat.html`

---

### 2. 统一部署拓扑与基础依赖
**开发内容**
- 推进 `postgres / qdrant / minio / api / inference` 这一套本地部署组合
- 统一主服务和独立推理服务的工作方式

**遇到的问题**
- 业务服务、推理服务、向量库、文件存储之间职责需要拆清楚

**处理方式**
- 保持主服务编排、推理服务独立、向量检索与文件存储分离

**涉及模块**
- `E:\project\chat-robot\docker-compose.yml`
- `E:\project\chat-robot\app\services\container.py`
- `E:\project\chat-robot\config\app.toml`

---

## 2026-04-21

### 1. 排查知识库导入卡住与资源占用异常

**开发内容**
- 排查上传知识库时页面无响应、GPU 占用升高到约 11G 的问题。
- 明确导入过程需要异步队列处理，避免上传请求长时间阻塞前端。
- 取消不必要的 health 心跳与无意义日志输出，减少页面和后端的背景噪声。
- 降低轮询频率，避免前端频繁请求进一步放大后端压力。

**遇到的问题**
- 小文件上传后长时间没有显示成功，用户无法判断任务是在排队、解析、切分还是写入向量库。
- 日志里存在大量 200、health 等低价值输出，真正异常被淹没。
- 上传任务和模型推理共享资源时，容易出现显存占用高、响应慢的问题。

**处理方式**
- 将导入流程拆成上传、解析、切分、embedding、写入向量库、写入关系库等阶段。
- 讨论并确认导入进度应来自后端任务状态，而不是单纯实时扫 Qdrant。
- 明确前端只展示有意义的任务状态和关键日志，不再展示无效心跳类信息。

**涉及模块**
- `api`
- `knowledge ingest`
- `admin`
- `frontend polling`
- `logging`

---

### 2. 评估模型缓存、显存占用与 vLLM 方案

**开发内容**
- 评估大模型启动即常驻缓存、BGE-M3 embedding 模型单例缓存的可行性。
- 讨论其余小模型是否适合加载时缓存进内存，以及对显存和系统内存的影响。
- 基于启动后仍有约 20G 系统内存剩余的现状，倾向将路由、embedding 等模型缓存起来提升响应速度。
- 讨论从 Transformers 推理切换到 vLLM 后，对首 token、吞吐和并发响应速度的可能提升。

**遇到的问题**
- 大模型常驻后显存仍有限，reranker、embedding、导入任务同时运行时可能互相挤占资源。
- vLLM 对生成链路有效，但不能直接解决文档解析、embedding、Qdrant 写入和前端进度展示问题。

**处理方式**
- 大模型作为主生成模型常驻。
- BGE-M3 按单例缓存处理，避免重复加载。
- vLLM 作为后续推理优化方向评估，但不把它当成导入卡顿问题的唯一解法。

**涉及模块**
- `model runtime`
- `embedding`
- `intent router`
- `llm generation`
- `vllm evaluation`

---

## 2026-04-22

### 1. 修复知识库上传反馈缺失与导入进度展示

**开发内容**
- 排查两个约 6MB / 3MB 文档上传后长时间无反馈的问题。
- 评估正常 3MB 文档解析、切分、embedding、写库的预期耗时。
- 调大 chunk size，优先提升导入速度，减少 chunk 数量和 embedding 次数。
- 将上传进度展示从页面临时状态调整为后端任务状态查询。

**遇到的问题**
- 页面只显示排队或无响应，用户无法看到已有任务进度。
- 切换知识库页面后，之前未完成的导入进度丢失。
- 前端轮询频率过高，但展示信息不足。

**处理方式**
- 明确进度数据来自导入任务记录，包括当前文档、已处理 chunk、总 chunk、阶段状态。
- 页面重新进入时查询未完成任务，恢复显示历史导入进度。
- 降低轮询频率，只在任务未完成时刷新。

**涉及模块**
- `knowledge ingest`
- `admin knowledge page`
- `task progress`
- `chunk config`

---

### 2. 修复知识库前端列表与多文件导入体验

**开发内容**
- 修复知识库页面报错 `document.createElement is not a function` 导致“已入库文档”无法展示的问题。
- 支持多文件批量导入，当前导入接口仍走 `/api/v1/knowledge/ingest`。
- 优化导入页的文件格式提示，为后续 doc、docx、pdf、txt、md 等格式支持预留入口。

**遇到的问题**
- 上传成功与入库成功没有清晰区分。
- 用户上传文件后看不到已入库文档，容易误判为文件丢失。
- 批量导入时缺少每个文件的独立状态展示。

**处理方式**
- 拆分“上传已接收”和“知识库已入库”的状态语义。
- 前端文档列表改为从后端查询真实记录。
- 导入任务按文件展示状态，避免多个文件混在一个模糊进度里。

**涉及模块**
- `admin knowledge page`
- `frontend document list`
- `knowledge api`
- `multi-file upload`

---

## 2026-04-23

### 1. 增强 RAG 后台管理与评测能力
**开发内容**
- 推进 RAG 管理后台、评测实验台和对比能力
- 支持对不同检索配置做对照测试

**遇到的问题**
- 没有实验台和评测能力时，调 retrieval / rerank 只能靠手工反复试
- 很难稳定比较配置变化到底有没有提升

**处理方式**
- 增加后台 RAG 配置页和评测入口
- 后续复杂题调优开始有统一入口

**涉及模块**
- `E:\project\chat-robot\app\api\routes\admin.py`
- `E:\project\chat-robot\app\static\admin-rag.html`
- `E:\project\chat-robot\app\static\js\admin-rag.js`

---

### 2. 升级对话页流式返回与打字机体验

**开发内容**
- 将前端对话页从“等待完整回答后一次性显示”调整为支持流式展示。
- 要求效果接近正常打字机输出，而不是最后一次性蹦出完整内容。
- 后端同步升级为支持流式输出，前端根据接口能力判断是否走流式模式。
- 将“进入队列等待多少秒”改为更符合用户体感的“正在思考多少秒”。

**遇到的问题**
- 仅前端模拟流式不能解决真实首 token 等待问题。
- 如果后端仍一次性返回，前端无法做到真正的 token 级流式体验。
- 用户体感慢主要来自意图识别、检索、重排、prompt prefill 和首 token 生成链路。

**处理方式**
- 前后端同时支持流式协议。
- 页面在请求发出后立即进入思考态，首段内容返回后切换为打字机展示。
- vLLM 继续作为首 token 和吞吐优化方向评估。

**涉及模块**
- `chat frontend`
- `streaming api`
- `llm generation`
- `console ui`

---

### 3. 改造后台为管理系统式导航结构

**开发内容**
- 将日志页面升级为后台管理系统，不再把所有功能挤在一个页面。
- 后台和本地对话控制台统一为科技感更强的视觉风格。
- 支持左侧或顶部导航，按系统概览、知识库、RAG 配置、流程日志、效果对比、性能统计等页面分区。
- 命中来源展示改为分页和懒加载，避免来源内容一直向下堆叠。

**遇到的问题**
- 原页面 card 过多，不同功能混在一起，页面可读性差。
- 长文本、来源列表、日志 JSON 容易溢出容器。
- 切换页面后部分状态没有恢复。

**处理方式**
- 按功能边界拆分管理页面。
- 长内容统一增加折叠、分页、懒加载和按需展开。
- 后台与控制台复用统一导航和视觉语言。

**涉及模块**
- `admin layout`
- `console layout`
- `source viewer`
- `frontend styles`

---

### 4. 增加知识库删除与向量库同步清理

**开发内容**
- 增加知识库文档删除功能。
- 强调删除必须同时清理关系库和 Qdrant 向量库，避免页面文档消失但向量仍被检索到。
- 排查用户上传的《神雕侠侣》在知识库页面消失的原因。
- 说明导入进度不是简单实时查询 Qdrant，而应来自导入任务记录。

**遇到的问题**
- 只删除关系库会造成幽灵向量，后续 RAG 仍可能命中已删除文档。
- 只查 Qdrant 难以表达解析中、embedding 中、写入失败、部分成功等状态。

**处理方式**
- 删除操作以文档 ID 为主键，同步删除文档记录、chunk 记录和 Qdrant points。
- 导入任务保留当前文件名、chunk 进度、状态和错误信息。
- 前端从任务接口恢复未完成导入进度。

**涉及模块**
- `knowledge delete`
- `qdrant`
- `relational db`
- `ingest progress`

---

## 2026-04-24

### 1. 完善 RAG 质量评测页面

**开发内容**
- 在 RAG 评测页面增加说明文案，包括问题集格式、参数含义和开始测评后的交互提示。
- 将评测参数明确为 chunk 大小、重复率、召回数量、重排数量、温度等。
- 支持用户输入问题和检索文件，对不同参数组合下的返回结果做页面简略展示。
- 评测结果暂存内存，不入库，避免测试数据污染正式知识库。

**遇到的问题**
- 原评测页缺少说明，用户不知道问题集如何填写，也不知道参数影响什么。
- 全量结果直接展示会挤爆页面。
- 不同参数下的结果需要对比，但不适合直接写入正式库。

**处理方式**
- 页面展示摘要结果，提供查看详情和导出 Excel / Word 的入口。
- 支持用户选择某组评测参数作为正式入库配置。
- 多参数结果只在当前进程内存保留。

**涉及模块**
- `admin compare`
- `rag evaluation`
- `export`
- `runtime config`

---

### 2. 增强评测文件输入格式

**开发内容**
- 评估并补充多格式文件输入能力。
- 将 txt、md、pdf、docx、doc 等格式纳入导入与评测入口的支持范围说明。
- 明确老格式 `.doc` 需要额外转换能力，例如 LibreOffice、antiword 或 Windows 组件。

**遇到的问题**
- 前端文件选择默认格式不包含 doc，用户容易误以为不支持。
- 不同文件格式解析链路不同，失败提示需要更清晰。

**处理方式**
- 前端补充支持格式提示和 accept 配置。
- 后端按文件扩展名走不同解析器，解析失败时返回明确错误。

**涉及模块**
- `file parser`
- `knowledge ingest`
- `admin compare`
- `frontend upload`

---

## 2026-04-25

### 1. 设计并接入 BM25 + 向量双检索

**开发内容**
- 梳理当前检索链路，原有主路径以 embedding 向量召回为核心。
- 设计接入 BM25 关键词检索，与 dense 向量检索形成 hybrid 双检索。
- 讨论为什么使用 SQLite FTS5 + jieba：本地部署简单、无需额外搜索服务、适合中文分词和轻量场景。
- 明确检索相关 5 个参数，包括 dense top_k、BM25 top_k、融合 top_k、重排 top_k、低分过滤阈值等。

**遇到的问题**
- 单纯向量检索对专有名词、编号、精确短语可能不稳定。
- 单纯 BM25 对语义改写不够鲁棒。
- 双检索结果需要融合、去重和过滤，否则会把重复 chunk 放大。

**处理方式**
- 增加 BM25 索引和查询能力。
- 使用 RRF 对 dense 和 BM25 结果做融合排序。
- 在进入重排前增加轻量去重和 top 截断。

**涉及模块**
- `retrieval`
- `bm25`
- `sqlite fts5`
- `jieba`
- `rrf`

---

### 2. 实现检索结果治理与引用编号

**开发内容**
- 增加同文档去重、同文档向量相似度去重、相邻 chunk 合并和引用编号映射。
- 用户指出正确顺序应为“同文档向量去重 -> 合并相邻 Chunk”，避免合并后的 chunk 没有现成向量而无法继续计算相似度。
- 模型基于知识库生成内容时，在回答后追加类似【1】的来源标注。

**遇到的问题**
- 如果先合并 chunk，就需要重新 embedding 才能继续做余弦相似度去重。
- 重复 chunk 会浪费重排和生成上下文 token。
- 来源编号如果不稳定，用户无法对应回答和命中来源。

**处理方式**
- 保持原始 chunk 向量用于相似度判断。
- 先做同文档向量去重，再做相邻 chunk 合并。
- 最终上下文生成引用编号，并在回答和来源列表之间建立映射。

**涉及模块**
- `retrieval governance`
- `dedupe`
- `chunk merge`
- `citations`
- `rag prompt`

---

### 3. 增加性能统计面板与导入队列交互

**开发内容**
- 基于已有 trace 数据实现极简性能统计页面。
- 单请求只展示意图判断、embedding、Qdrant 检索、重排、首 token 生成、总耗时 6 个关键数字。
- 排查知识库导入页面一直显示排队的问题。
- 评估导入队列内容是否支持取消。

**遇到的问题**
- 原 trace 数据过多，不利于快速定位慢在哪一环。
- 队列中已有任务时，新任务只显示排队，但页面没有展示前置任务进度。
- 重排耗时明显偏高，且低 score 内容仍被返回。

**处理方式**
- 性能页用表格展示单请求核心耗时。
- 导入页显示当前运行任务和排队任务。
- 后续为队列任务补充取消能力和状态刷新。

**涉及模块**
- `admin performance`
- `trace`
- `knowledge queue`
- `reranker`

---

## 2026-04-26

### 1. 优化重排耗时与前置粗去重

**开发内容**
- 评估 reranker GPU 分时复用、CPU 推理优化、ONNX INT8、批量推理等方案。
- 最终先按用户要求只做“前置粗去重”，不引入 GPU / CPU 推理改造。
- 将召回后的候选先按 document_id 分组，同文档优先保留检索分数最高的少量 chunk。
- 增加完全重复文本过滤，减少进入重排模型的候选数量。

**遇到的问题**
- 原流程中大量冗余 chunk 被送入 reranker，重排耗时接近线性增长。
- 召回 50 条直接重排会造成明显延迟。
- 低分 chunk 进入最终上下文会污染回答质量。

**处理方式**
- 调整为“召回 -> 轻量前置粗去重 -> top 截断 -> 重排 -> 精细后处理合并”。
- 对话页增加是否重排开关，便于在低延迟和高质量之间切换。
- 低分过滤不依赖某一种模式，RRF 和 rerank 路径都应执行。

**涉及模块**
- `reranker`
- `retrieval prefilter`
- `chat settings`
- `rag config`

---

### 2. 梳理意图识别、follow_up 与首 token 延迟

**开发内容**
- 解释当意图识别为 follow_up 时，系统不一定会重新走完整 RAG，应结合上下文判断是否需要知识库。
- 排查个别情况下意图识别耗时高和首 token 生成慢的原因。
- 讨论规则引擎增强、轻量模型缓存、上下文 token 限制、vLLM / TGI 等优化路径。

**遇到的问题**
- 意图识别如果仍走大模型或复杂模型，偶发延迟会很高。
- RAG 上下文过长会增加 prefill 时间，拖慢首 token。
- 用户体感上，首 token 延迟比总耗时更明显。

**处理方式**
- 后续优先增加意图规则配置和缓存。
- RAG 配置中保留最大上下文 token 限制方向。
- vLLM 继续作为生成链路优化项，但不替代检索治理。

**涉及模块**
- `intent`
- `follow_up`
- `first token latency`
- `rag config`
- `llm runtime`

---

## 2026-04-27

### 1. 增强 trace 与评估观测
**开发内容**
- 推进 trace、评估与后台观测能力
- 为后续复杂执行链路可视化和回归做准备

**遇到的问题**
- 没有细粒度 trace 时，复杂问题链路一旦变长，就无法快速判断问题出在：
  - 路由
  - 检索
  - 生成
  - 还是后续聚合

**处理方式**
- 补强 trace 结构
- 为后续 planner、tool selection、KB selection 等步骤接入观测打底

**涉及模块**
- `E:\project\chat-robot\app\services\trace_service.py`
- `E:\project\chat-robot\app\static\admin-traces.html`
- `E:\project\chat-robot\app\static\js\admin-traces.js`

---

### 2. 总结 RAG 当前能力与后台管理缺口

**开发内容**
- 汇总当前系统已实现能力，包括知识库导入、向量检索、BM25、RRF、rerank、来源标注、流式输出、后台配置和流程 trace。
- 评估后续优先级，包括 RAG 质量评测、检索结果治理、性能统计面板、首 token 延迟优化。
- 明确后台管理需要同时支持单次流程数据日志、RAG 参数实时调整、chunk 切割参数、是否重排、不同流程结果对比。
- 要求流程数据默认懒加载，只在点击查看全部时展示完整内容，并支持分页。

**遇到的问题**
- 原日志页只像调试页面，不像后台管理系统。
- RAG 参数、评测结果、流程日志、性能耗时分散，调参成本高。
- 全量日志一次性渲染会拖慢页面。

**处理方式**
- 将后台拆为管理系统式页面。
- RAG 配置页展示当前配置，并支持实时生效。
- 流程日志和评测结果按分页、摘要、详情展开的方式组织。

**涉及模块**
- `admin`
- `rag config`
- `flow logs`
- `compare`
- `trace`

---

## 2026-05-07

### 1. 收口复杂问题处理策略
**开发内容**
- 讨论并明确复杂问题不继续走自由 ReAct，而收口到受控执行模式
- 将复杂问题统一归到 `plan_execute`

**遇到的问题**
- 存在多种可能方向：
  - 直接做完整 Skill 体系
  - 让小模型硬扛 ReAct
  - 普通 RAG 上继续堆 prompt
- 这些路线都容易让当前模型和项目阶段负担过大

**处理方式**
- 明确：
  - 小模型负责轻路由、轻分类
  - 8B 更适合做 planner 和最终回答
- 将复杂问题正式收口到：
  - `direct`
  - `rag`
  - `plan_execute`
  - `clarify`

**涉及模块**
- `E:\project\chat-robot\app\services\intent_service.py`
- `E:\project\chat-robot\app\schemas\chat.py`

---

### 2. 明确 planner 与执行层职责边界
**开发内容**
- 重新定义 planner 和 executor 的边界
- planner 只拆任务，不承担完整工具编排
- 执行层负责：
  - tool selection
  - knowledge base selection
  - task intent
  - runtime execute

**遇到的问题**
- 早期 planner 设计过重，容易再次把所有复杂度堆回 prompt

**处理方式**
- 明确职责拆分
- 为第二天的主实现铺路

**涉及模块**
- `E:\project\chat-robot\app\services\planner_service.py`
- `E:\project\chat-robot\app\services\chat_pipeline.py`

---

## 2026-05-08

### 1. 收口复杂问题执行主链路
**开发内容**
- 将复杂问题处理链路正式整理为：
  - `planner`
  - `task`
  - `tool_selection`
  - `knowledge_base_selection`
  - `runtime execute`
  - `subtask completion`
  - `aggregate`
  - `final answer`
- 保留 `execution_strategy = off / auto / force`

**遇到的问题**
- 复杂问题早期仍然容易被普通 `rag` 吃掉
- `react` 语义不够清晰，不利于后续演进

**处理方式**
- 正式使用 `plan_execute`
- 保留旧字段兼容，但统一执行新链路

**涉及模块**
- `E:\project\chat-robot\app\services\chat_pipeline.py`
- `E:\project\chat-robot\app\schemas\chat.py`

---

### 2. 收轻 planner 职责并提升稳定性
**开发内容**
- 将 planner 收轻为“任务拆解器”
- 精简 planner prompt
- 将 planner 默认 `max_new_tokens` 调整为 `512`
- 增强 planner fallback

**遇到的问题**
- planner 早期 prompt 太重
- 8B 经常：
  - 自由解释
  - 只输出半截 JSON
  - 自由分析任务而不是直接吐结构

**处理方式**
- 精简输出要求
- 收缩 planner 职责
- 调大输出预算
- 保留 guardrail 兜底

**涉及模块**
- `E:\project\chat-robot\app\services\planner_service.py`
- `E:\project\chat-robot\config\models.toml`
- `E:\project\chat-robot\app\core\config.py`

---

### 3. 增加任务依赖与 DAG-lite 调度
**开发内容**
- 为 `PlanTask` 增加 `depends_on`
- 执行层支持按拓扑层分批执行
- 同层任务并发，依赖任务等待前置结果

**遇到的问题**
- 复杂任务不是所有 task 都能并行
- 例如比较、交集这类任务必须等上游结果先完成

**处理方式**
- 用 DAG-lite 代替“全量同时执行”
- 下游任务通过 `dependency_results` 获取上游结果

**涉及模块**
- `E:\project\chat-robot\app\schemas\chat.py`
- `E:\project\chat-robot\app\services\planner_service.py`
- `E:\project\chat-robot\app\services\chat_pipeline.py`

---

### 4. 新增 Tool Runtime 第一版
**开发内容**
- 增加统一工具运行时
- 注册默认工具：
  - `retrieval.search`
  - `answer.direct`
  - `kb.document_lookup`
  - `trace.lookup`
- 后台支持列出 runtime tools

**遇到的问题**
- 执行层之前还是大量硬编码
- 后续如果要做 Skill、MCP、function calling，没有统一入口

**处理方式**
- 引入 `tool_runtime.py`
- 执行层开始通过 runtime 统一调工具

**涉及模块**
- `E:\project\chat-robot\app\services\tool_runtime.py`
- `E:\project\chat-robot\app\services\container.py`
- `E:\project\chat-robot\app\api\routes\admin.py`

---

### 5. 新增 task 级工具选择
**开发内容**
- 每个 task 执行前先做一次轻量 `tool_selection`
- 模型输出：
  - `tool`
  - `arguments`
  - `reason`
- 执行层保留 fallback

**遇到的问题**
- planner 收轻之后，后续任务如何执行需要另行决策
- 如果继续在执行层写死，Tool Runtime 的价值发挥不出来

**处理方式**
- 增加 `task -> tool_selection -> runtime execute`

**涉及模块**
- `E:\project\chat-robot\app\services\chat_pipeline.py`
- `E:\project\chat-robot\app\services\generator_service.py`

---

### 6. 新增 task 级知识库选择
**开发内容**
- 在 `tool_selection` 后增加 `knowledge_base_selection`
- 只对需要知识库的工具触发
- 单知识库直接走默认库

**遇到的问题**
- 不是所有 task 都需要选知识库
- 过早全局选库会限制复杂多任务演进

**处理方式**
- 给工具定义加 `requires_knowledge_base`
- 仅检索类工具触发 KB selection

**涉及模块**
- `E:\project\chat-robot\app\services\tool_runtime.py`
- `E:\project\chat-robot\app\services\chat_pipeline.py`

---

### 7. 新增 task 级意图识别
**开发内容**
- 每个 task 执行前再做一次轻量意图识别
- 支持：
  - `direct`
  - `retrieval`
  - `extraction`
  - `aggregation`
- `extraction` 任务额外抽取：
  - `query`
  - `source_hint`
  - `target`
  - 可选 `knowledge_base_id`

**遇到的问题**
- 不是所有 task 都该直接按普通检索执行
- “列出/统计/共同项”更像抽取任务

**处理方式**
- 在执行层插入 `task_intent`
- 让 `extraction` 先收紧任务查询条件，再进入工具和知识库选择

**涉及模块**
- `E:\project\chat-robot\app\services\chat_pipeline.py`
- `E:\project\chat-robot\app\services\generator_service.py`

---

### 8. 增强聚合器和执行质量控制
**开发内容**
- 聚合器支持：
  - `union`
  - `intersection`
  - `compare`
  - `dedupe_union`
  - `rank`
  - `group_by`
- 增加 `SubtaskResult` 和 `AggregateResult` 质量字段
- 增加一次补检索
- 增加候选项清洗和低质量回退

**遇到的问题**
- 没有统一聚合器时，复杂问题只能靠最终模型自行归纳
- 复杂任务没有质量判断时，很难知道哪一步失真

**处理方式**
- 引入统一聚合器和结果质量字段

**涉及模块**
- `E:\project\chat-robot\app\schemas\chat.py`
- `E:\project\chat-robot\app\services\chat_pipeline.py`

---

### 9. 增强 Trace、后台与评测能力
**开发内容**
- 前端显示 planner 和 execution steps
- 后台支持查看 runtime tools
- 后台增加默认复杂题评测集
- trace 补充 planner、tool selection、knowledge base selection 等步骤

**遇到的问题**
- 没有中间可视化时，很难定位问题出在：
  - planner
  - tool selection
  - knowledge base selection
  - 检索
  - 聚合
  - 还是最终生成

**处理方式**
- 将关键步骤接入 trace 和后台

**涉及模块**
- `E:\project\chat-robot\app\api\routes\admin.py`
- `E:\project\chat-robot\app\static\admin-rag.html`
- `E:\project\chat-robot\app\static\js\admin-rag.js`
- `E:\project\chat-robot\app\static\admin-traces.html`
- `E:\project\chat-robot\app\static\js\admin-traces.js`
- `E:\project\chat-robot\app\static\chat.html`
- `E:\project\chat-robot\app\static\js\chat-console.js`

---

### 10. 排查 planner 输入是否乱码
**开发内容**
- 单独验证 planner 输入在进入模型前是否已经乱码
- 用 Unicode 安全方式重放请求

**遇到的问题**
- 早期看到 planner 输出中存在 `????`

**处理方式**
- 使用 Unicode 转义和 UTF-8 安全请求重放
- 确认真实业务链路中的输入并未系统性乱码
- 判断部分问号现象主要来自 shell 传参与显示编码

**涉及模块**
- `E:\project\chat-robot\app\services\planner_service.py`
- `E:\project\chat-robot\app\services\rag_snapshot_service.py`

---

### 11. 验证 planner 输出截断与 token 关系
**开发内容**
- 比较不同 `max_new_tokens` 下 planner 输出变化

**遇到的问题**
- planner 经常只输出半截 JSON

**处理方式**
- 验证 `128 / 256 / 512` 三档输出
- 确认 `256` 较容易截断
- 将 planner 默认输出预算调高
- 同时判断问题不只是 token，而是 prompt 过重与自由 JSON 约束不足

**涉及模块**
- `E:\project\chat-robot\config\models.toml`
- `E:\project\chat-robot\app\core\config.py`

---

### 12. 验证复杂问题真实执行链路
**开发内容**
- 多次用真实服务验证“射雕 / 神雕共同武功”与相关变体问题
- 验证：
  - planner
  - tool selection
  - knowledge base selection
  - task intent
  - aggregate
  
  是否真实在线生效

**遇到的问题**
- 虽然链路开始走对，但最终结果仍然不稳定
- 候选项中大量混入描述性片段，而不是稳定实体名

**处理方式**
- 逐步缩小问题边界
- 确认当前主要问题已经不是链路没接上，而是 extraction 任务的结果质量

**涉及模块**
- `E:\project\chat-robot\app\services\chat_pipeline.py`

---

### 13. 验证“是不是模型太小导致检索烂”
**开发内容**
- 使用 `/api/v1/admin/rag/compare`
- 设置 `generate_answer = false`
- 只看 sources，不让最终回答模型参与

**遇到的问题**
- 怀疑问题可能主要来自本地模型太小

**处理方式**
- 将检索与回答模型影响拆开验证

**结论**
- 纯检索阶段本身就已经存在明显偏差
- 问题不能简单归因为“8B 太小”

**涉及模块**
- `E:\project\chat-robot\app\api\routes\admin.py`
- `E:\project\chat-robot\app\services\retriever_service.py`

---

### 14. 验证 BM25 / 双检索偏差
**开发内容**
- 对同一查询分别跑：
  - `dense`
  - `bm25`
  - `hybrid`
- 观察不同模式下的 sources

**遇到的问题**
- 怀疑双检索中的词项匹配加剧了 `武功 / 功夫` 这类泛词噪声

**处理方式**
- 使用纯检索对照方式逐项验证

**结论**
- BM25 / 词项匹配确实会放大偏差
- 尤其对泛词 query 容易抬高“只是提到了武功/功夫”的叙述片段
- 但 dense 本身也没有彻底解决问题
- 根因仍然是：当前用普通相关性 top-k 检索在做实体枚举任务

**涉及模块**
- `E:\project\chat-robot\app\services\retriever_service.py`
- `E:\project\chat-robot\app\api\routes\admin.py`

---

### 15. 评估动态加权方案
**开发内容**
- 讨论是否根据用户问题动态调整检索权重
- 评估是否由模型直接输出动态权重系数

**遇到的问题**
- 当前固定 hybrid 参数无法很好覆盖不同类型任务
- 不同问题对：
  - dense
  - bm25
  - rerank
  
  的需求不同

**处理方式**
- 判断动态加权方向可行
- 但不建议第一版让模型直接输出连续权重
- 更推荐模型先判断 `retrieval_profile`，再由执行层映射固定参数档位

**当前结论**
- 动态加权值得做
- 第一版更适合做“动态选档”，而不是“模型自由输出连续权重”

---

### 16. 统一 RRF、重排、低分过滤和前置去重逻辑

**开发内容**
- 梳理 RRF 和 rerank 两条路径的真实执行逻辑。
- 明确无论走 RRF 还是重排，低分过滤和前置粗去重都应该执行。
- 检查 hybrid 模式是否已经做去重和过滤，避免同一批 chunk 在不同阶段重复做无效处理。
- 在重排前增加 top 截断，避免把所有召回结果都送入 reranker。

**遇到的问题**
- 如果 RRF 路径和 rerank 路径过滤逻辑不一致，会导致“关闭重排后质量突然变差”。
- 粗去重和低分过滤如果分散在多个阶段，容易出现重复计算或遗漏。
- 不截断直接重排会造成明显性能浪费。

**处理方式**
- 抽象统一候选治理链路：召回结果先做低分过滤、文本去重、同文档粗去重，再进入 RRF 或 rerank 后续阶段。
- RRF 作为低延迟排序路径，rerank 作为高质量可选路径。
- 后台和对话页保留是否重排开关，便于快速对比。

**涉及模块**
- `retrieval pipeline`
- `rrf`
- `reranker`
- `rag config`
- `chat settings`

---

### 17. 调整导航命名与效果对比入口

**开发内容**
- 按用户反馈调整导航展示，不再把“效果对比”和“RAG 评测”混成同一个含义。
- 对话页面导航栏展示“效果对比”，后台其他页面展示“RAG 评测”。
- 保持后台页面结构统一，避免不同入口显示不一致。

**遇到的问题**
- 页面命名不清会导致用户不知道当前功能是在线对话对比，还是离线 RAG 参数评测。
- 导航项如果在不同页面语义漂移，会降低后台可用性。

**处理方式**
- 将导航文案按使用场景区分。
- 页面功能保持拆分，不把无直接关系的内容放在一个页面里。

**涉及模块**
- `admin navigation`
- `console navigation`
- `compare page`

---

### 18. 增加内存级全链路快照与单请求诊断详情页

**开发内容**
- 评估并实现全链路快照留存，数据保存在内存中，不做持久化存储。
- 快照覆盖用户 query、意图识别结果、召回 chunk、RRF 融合前后排序、重排前后排序、最终 prompt、LLM 原始输出和后处理结果。
- 增加单请求详情页，用于人工回溯一次请求的召回、排序和生成链路。
- 不做自动诊断规则，因为正确与否需要人工判断。

**遇到的问题**
- 只有耗时 trace 时，能知道慢在哪，但不知道错在哪。
- 全量 JSON 直接展示会让页面卡顿，也不利于阅读。
- 快照如果持久化会引入存储膨胀和隐私问题，当前项目阶段没有必要。

**处理方式**
- 只在内存保留最近请求快照。
- 首页展示中文处理后的关键字段。
- 原始 JSON 放到“查看 JSON”按钮里按需展开。

**涉及模块**
- `trace snapshot`
- `admin trace detail`
- `retrieval diagnostics`
- `llm prompt inspection`

---

### 19. 优化诊断详情页中文展示与长内容懒加载

**开发内容**
- 将检索链路中的英文/内部字段改为中文标题和中文 label。
- 对原始详情、prompt、chunk 文本、排序列表等长内容做懒加载和折叠。
- 移除默认页面上没有展示意义的内部 JSON 字段。
- 修复灰底白字、墨蓝色文字等低对比度样式问题。

**遇到的问题**
- 原详情页直接暴露内部结构，用户不知道字段代表什么。
- 页面一次性渲染过多文本会变卡。
- 低对比度配色导致内容看不清。

**处理方式**
- 默认只展示人工排查真正需要看的摘要。
- 长文本和原始 JSON 改为点击后再渲染。
- 调整背景、文字和边框颜色，保证可读性优先。

**涉及模块**
- `admin trace detail`
- `frontend lazy render`
- `trace ui`
- `styles`

---

### 20. 评估并补充知识库分库与检索范围控制

**开发内容**
- 发现当前 RAG 缺少分库能力，不能清晰区分不同知识库。
- 评估在意图识别阶段判断应调用哪个知识库，再限定后续检索范围。
- 规划知识库元数据、上传归属、Qdrant filter、BM25 filter 和轻量路由逻辑。

**遇到的问题**
- 不分库会让小说、业务文档、技术资料互相污染召回结果。
- 用户需要按知识库管理文档，也需要按知识库控制检索范围。
- 历史已导入文档可能需要迁移到默认知识库或重新标注归属。

**处理方式**
- 将知识库作为一层独立管理对象。
- 文档、chunk、向量 payload 和 BM25 索引都带知识库 ID。
- 意图识别或轻量路由先选库，再执行 dense / BM25 / RRF / rerank。

**涉及模块**
- `knowledge base`
- `document metadata`
- `qdrant filter`
- `bm25 filter`
- `intent routing`

---

## 当前阶段结论

### 已经完成
- 复杂问题的 `plan -> task -> execute -> aggregate` 主链路搭建
- Tool Runtime 第一版
- task 级工具选择
- task 级知识库选择
- task 级意图识别
- DAG-lite 任务依赖调度
- 聚合器扩展
- trace / 后台 / 评测增强
- 知识库导入队列化、进度恢复和删除同步
- BM25 + dense 双检索、RRF 融合、重排开关和检索结果治理
- 内存级全链路快照与单请求诊断详情页
- 知识库分库与检索范围控制的基础设计

### 当前主要问题
- `extraction` 子任务的结构化抽取稳定性不足
- 小说类“实体枚举 / 共同项”任务不适合继续只靠普通 top-k 检索
- `武功 / 功夫` 这类泛词会放大 BM25 噪声
- CPU reranker 在候选较多时仍然容易成为主要耗时点
- 历史文档需要确认是否迁移到默认知识库或重新标注归属

### 当前判断
- 当前主问题已经不是链路没搭起来
- 也不只是 planner 不会拆任务
- 现在真正卡效果的是：
  1. extraction 任务的执行范式
  2. extraction 的结构化输出稳定性
  3. retrieval profile 是否应该动态调整
  4. 知识库路由和分库检索是否足够稳定

---

