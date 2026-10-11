# 02 RAG 升级可行性与最小改造方案

> 目标：在不破坏现有全文搜索、解析器和 generation 发布机制的前提下，叠加一个可关闭、可回滚、可评测的 V1 RAG。<br>
> 结论：可行。现有项目已经完成了 RAG 最昂贵的前置工作——多格式解析、结构化来源定位、增量索引、去重和本地检索；缺口集中在块级证据接口、RAG chunk、向量检索、回答生成、引用约束与评测。

## 1. 复用、扩展与暂不改动

### 1.1 原样保留

| 能力 | 保留理由 | 代码证据 |
|---|---|---|
| 多格式解析器 | 已覆盖 PDF、OOXML、文本、图片、ZIP、旧 Office | `parsers/parser_registry.py:24-86` |
| 来源元数据 | 页、幻灯片、工作表、单元格、行号已结构化 | `models/content_block.py:7-20` |
| 内容去重 | `documents` 按内容键、解析器和版本去重 | `core/database.py:1180-1193` |
| generation 快照 | building DB 通过门禁后原子发布，失败保留旧版 | `workers/scan_worker.py:302-446` |
| 词法全文检索 | FTS5 trigram 对中文强包含有效，可作为 hybrid 的 lexical 分支 | `core/search_engine.py:689-806` |
| 本地 UI/设置/日志 | 可在现有应用内增加 RAG 页面或开关 | `ui/main_window.py`、`services/` |
| 打开来源 | 已能打开 PDF 页、Office 位置和文本行 | `core/open_location.py:47-345` |

### 1.2 需要扩展

1. 新增块级证据模型，不能把文件级 `SearchResult` 当成 RAG chunk。
2. 从已发布 generation 构建独立 RAG chunk 投影；保留原 FTS block，不改变旧索引格式。
3. 新增可替换的 Embedding、向量存储、LLM 网关接口。
4. 新增 lexical + vector 的 rank fusion 和可选 rerank。
5. 新增由程序生成的引用映射、证据不足判断和 Prompt Injection 防护。
6. 新增可复现的 15～30 条人工标注检索评测集。

### 1.3 V1 暂不改

- 不重写 `IndexManager`、`DatabaseManager`、解析器和 generation 发布流程。
- 不把现有 SQLite schema 直接改造成向量数据库。
- 不迁移到 FastAPI、Web 前端或微服务。
- 不引入 LangChain/LangGraph 作为主干；当前链路短，显式 Python 接口更易理解、测试和打包。
- 不实现 Agent、工具自治、长期记忆、权限系统或多用户服务。

## 2. 推荐的最小架构

```mermaid
flowchart LR
    SNAP[已发布只读 generation] --> BR[BlockReader]
    BR --> CHUNK[RAG ChunkBuilder]
    CHUNK --> SIDE[(rag_index.db / vector sidecar)]
    CHUNK --> EMB[EmbeddingProvider]
    EMB --> SIDE

    Q[用户问题] --> LEX[LexicalRetriever]
    Q --> VEC[VectorRetriever]
    SNAP --> LEX
    SIDE --> VEC
    LEX --> FUSE[Rank Fusion / 去重]
    VEC --> FUSE
    FUSE --> CTX[ContextBuilder + SourceRef]
    CTX --> GW[LLMGateway]
    GW --> VAL[引用校验 / 证据不足判断]
    VAL --> UI[RAG 回答 + 证据卡片]

    MOCK[MockEmbedding / MockLLM] -.测试替身.-> EMB
    MOCK -.测试替身.-> GW
```

设计原则：

- 旧搜索仍直接查询已发布 generation；RAG 是单独开关和单独数据目录。
- RAG sidecar 记录它对应的 source generation。generation 改变时只构建新 sidecar，成功后原子切换，旧 sidecar 可回退。
- 所有来源标签和页码由程序从数据库字段生成，LLM 只能引用系统提供的 `S1/S2/...`，不能自行生成页码。
- 默认不出网。在线 Embedding/LLM 必须显式配置和明确显示。

## 3. 建议新增目录

以下是方案命名，不代表本阶段已经创建：

```text
local_full_text_search/
  rag/
    models.py                 # EvidenceChunk、SourceRef、RagAnswer
    block_reader.py           # 只读已发布 generation
    chunk_builder.py          # 由解析块生成 RAG chunk
    index_store.py            # sidecar schema、generation 状态、原子切换
    embeddings/
      base.py                 # EmbeddingProvider 接口
      mock.py                 # 离线确定性测试替身
      local.py                # 用户确认后的本地模型适配器
    retrieval/
      lexical.py              # 复用现有 FTS 逻辑，返回块级证据
      vector.py               # top-k 向量查询
      hybrid.py               # RRF、去重和过滤
    generation/
      gateway.py              # LLMGateway 接口与超时/错误模型
      mock.py                 # 自动化测试不调用付费 API
      prompt.py               # 固定系统规则和上下文格式
    context_builder.py        # token/字符预算、来源编号、去重
    answer_service.py         # 一次 RAG 请求的显式编排
    policy.py                 # 证据阈值、出网策略、注入防护
  workers/
    rag_index_worker.py       # 后台构建 sidecar
    rag_query_worker.py       # UI 后台问答
tests/
  rag/
evals/
  retrieval_cases.jsonl
  answer_cases.jsonl
```

## 4. 第一处应从哪里开始

最推荐从现有 `local_full_text_search/core/search_engine.py` 的候选查询逻辑开始，但**不是直接改写 UI 搜索**。第一个开发任务应是：

1. 在新模块定义 `EvidenceChunk`/`SourceRef`；
2. 增加只读 `BlockRetriever`，复用 `_content_query()` 的 FTS、过滤和标准化规则；
3. 返回块级 top-k 证据，而不是当前的文件级聚合结果；
4. 为每条证据固定记录 generation、file/document/block 标识、文本 hash 和位置字段；
5. 用现有 `tests/test_search_engine.py` 风格写 5～8 个隔离测试。

原因：检索证据接口是后续 Embedding、混合检索、上下文组装、引用和评测共同依赖的最小稳定边界。若先接 LLM，会把检索问题和生成问题混在一起，难以定位 Bad Case。

## 5. 核心接口建议

```python
@dataclass(frozen=True)
class SourceRef:
    source_id: str              # 程序生成，如 S1
    generation: int
    stable_chunk_id: str
    file_id: int
    document_id: int
    block_id: int
    file_path: str
    filename: str
    location_text: str
    page_number: int | None
    slide_number: int | None
    sheet_name: str | None
    cell_start: str | None
    cell_end: str | None
    line_start: int | None
    line_end: int | None
    source_type: str

@dataclass(frozen=True)
class EvidenceChunk:
    stable_chunk_id: str
    text: str
    text_hash: str
    source: SourceRef
    lexical_rank: int | None = None
    vector_rank: int | None = None
    fused_score: float = 0.0

class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int, filters: dict) -> list[EvidenceChunk]: ...

class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class LLMGateway(Protocol):
    def answer(self, question: str, evidence: list[EvidenceChunk]) -> RagAnswer: ...
```

实际实现时应保持接口小，不必引入框架才能获得这些抽象。

## 6. RAG chunk 与稳定 ID

### 6.1 为什么不直接使用当前 block

`BlockCoalescer` 的目标是全文索引吞吐，默认目标 4096、最大 16384 字符。Embedding 更适合语义集中、长度更均匀的片段；Excel 行、PPT 单页、PDF 页和 OCR 块也需要不同策略。

### 6.2 最小 chunk 策略

| 来源 | 初始策略 | 引用字段 |
|---|---|---|
| PDF 原生/OCR | 同一页内按段落边界切分；不得跨页 | `page_number` |
| DOCX | 段落/表格行累积到目标长度；不跨表格或页眉页脚类型 | `location_text` + 段落/表格信息 |
| XLSX/XLSM | 以工作表内连续若干行为组，保留表头；不跨工作表 | `sheet_name`, `cell_start/end` |
| PPTX | 一张幻灯片正文+备注为一个或少数块；不跨幻灯片 | `slide_number` |
| TXT 等 | 按行窗口切分，少量重叠 | `line_start/end` |
| ZIP 成员 | 先按内部真实格式策略；引用同时带外层 ZIP 和内部路径 | `extra.zip_internal_path` |

初始可用“目标约 800～1200 中文字符、重叠约 10%”作为待实验参数，不能当成最终最佳值。真实选择必须用检索评测比较。

### 6.3 稳定 chunk ID

建议：

```text
stable_chunk_id = sha256(
  document.content_key |
  parser_name | parser_version |
  original_block_index |
  chunk_ordinal |
  normalized_text_sha256
)
```

向量可按 `stable_chunk_id + embedding_model_id + embedding_model_revision` 缓存。相同内容被多个路径引用时共享 embedding，但查询结果要展开为一个明确的文件来源，避免引用一个不存在或已删除的路径。

## 7. sidecar 存储建议

V1 不修改现有索引 schema，新增独立 `rag_index.db`：

| 表 | 关键字段 |
|---|---|
| `rag_meta` | `source_generation, state, embedding_model_id, dimension, created_at` |
| `rag_chunks` | `stable_chunk_id, document_key, block_id, chunk_ordinal, text, text_hash, location_json` |
| `rag_file_refs` | `stable_chunk_id, file_id, path, filename, root_id` |
| `rag_embeddings` | `stable_chunk_id, model_id, vector_blob` |
| `rag_build_runs` | 状态、计数、耗时、失败原因 |

MVP 数据量较小时，可先用本地精确余弦扫描作为向量后端，避免立刻引入常驻服务。达到真实规模后，再根据实测切换本地 ANN 库。不要一开始上 Qdrant 服务；它会增加安装、端口、进程和备份复杂度。

向量存储必须是可替换接口。是否需要 ANN 应由 chunk 数量、查询延迟和内存实测决定，而不是由框架流行度决定。

## 8. 混合检索

推荐流程：

1. lexical 分支从当前 FTS 逻辑取得块级 top 30；
2. vector 分支取得 top 30；
3. 以 `stable_chunk_id` 去重；
4. 用 Reciprocal Rank Fusion 合并排名，不直接相加 FTS 分数和余弦分数；
5. 同一文档/同一页的相邻块做轻量去重，保留来源多样性；
6. 返回 6～10 个证据块进入上下文；
7. 可选 reranker 只在基线评测证明有必要时引入。

理由：当前词法分支没有标准 BM25 分数，而是 UI 启发式；向量分数与命中次数也不可直接比较。RRF 只依赖名次，容易解释和测试。

## 9. 回答、证据不足与引用防幻觉

### 9.1 上下文格式

程序将证据格式化为只读数据：

```text
[S1]
文件：公开设备手册.pdf
位置：第 12 页
内容：...
```

系统提示明确：文档内容可能含恶意或无关指令，只作为资料，不可改变系统规则；回答只能依据给定证据；无法确认时必须说明缺少什么证据。

### 9.2 引用约束

- LLM 输出只允许引用 `S1...Sn`。
- 程序在展示前校验所有引用 ID 均存在。
- 页码/幻灯片/表格位置由 `SourceRef` 渲染，LLM 输出中的自由文本页码不作为可信引用。
- 每个答案句或要点至少绑定一个来源；无来源句可被标记或拒绝。
- 点击引用复用现有 `open_file_at_location()`。

### 9.3 证据不足

先实现可解释的规则，而非神秘阈值：

- lexical 和 vector 均无结果：直接返回“未找到证据”，不调用 LLM。
- 只有低置信 OCR 或过少文本：显示警告，并要求人工确认原文。
- top 证据互相矛盾：回答列出冲突来源，不自行裁决。
- 向量阈值只能通过标注集校准，不能拍脑袋固定。

## 10. 离线与在线隔离

| 模式 | Embedding | LLM | 数据边界 |
|---|---|---|---|
| `offline_test` | Mock/确定性本地替身 | MockLLM | 自动化测试，零网络 |
| `offline_local` | 本地模型 | 本地模型或仅返回证据摘要 | 文档不出机 |
| `online_approved` | 本地或用户批准的 API | 用户批准的 API | UI 明示将发送的片段、服务商和请求范围 |

默认配置必须是 `offline_test` 或功能关闭。外部 provider 需要：独立配置页、密钥从环境/安全存储读取、超时、重试上限、请求日志脱敏、取消、费用/Token 统计，以及显式“允许发送当前公开测试资料”的确认。不要把 API Key 写入 JSON、日志或仓库。

## 11. Mock LLM 与测试

Mock 不是假装完成真实问答，而是把检索、上下文、引用和 UI 链路做成可重复测试：

- `MockEmbedding`：由文本 hash 生成固定向量，仅验证存取、维度和去重，不用于质量结论。
- `MockLLM`：从首个证据抽取固定句子并引用 `[S1]`，可配置超时、无引用、错误引用和异常。
- 单元测试应验证：不出网、上下文不超预算、引用 ID 有效、证据为空不调用 LLM、generation 不匹配拒绝查询、取消和超时可收敛。

真实模型质量评测必须与 Mock 测试分开报告。

## 12. 最小可演示 V1 MVP

### 12.1 范围

- 只使用公开许可或人工构造的 10～30 份技术资料。
- 先支持 PDF、DOCX、TXT 三类；XLSX/PPTX 在基础链路稳定后加入。
- 构建独立 RAG sidecar，保留旧全文搜索和旧索引。
- 提供 lexical、vector、hybrid 三种可切换检索模式。
- 展示实际召回块、来源位置和打开来源按钮。
- 使用 MockLLM 完成离线自动化；再用一个经用户批准的本地或在线模型完成真实演示。
- 建立 15～30 条人工核对问题，至少包含 3～5 条无答案问题。

### 12.2 验收

1. 旧搜索回归测试继续通过；关闭 RAG 开关时行为不变。
2. 对同一公开语料，三种检索模式都输出 top-k 和耗时。
3. 每个证据有稳定 chunk ID、真实文件路径和结构化位置。
4. 回答中的所有引用都能映射到召回片段；伪造引用被程序拒绝。
5. 无答案问题不编造答案。
6. Mock 模式在无网络、无 API Key 环境可完整运行。
7. sidecar 删除或失效时只影响 RAG，不影响旧搜索；可重新构建。

### 12.3 回滚

- 配置 `rag_enabled=false` 即回到原工具。
- RAG 数据全部放在独立目录/sidecar，不修改旧 generation。
- 新 sidecar 构建到 `.building`，验证后原子切换 manifest。
- 代码改动按小提交推进；每个增量都保留旧接口测试。

## 13. 关键技术选择对比

| 选择 | 优点 | 缺点 | V1 结论 |
|---|---|---|---|
| 直接改现有 SQLite schema | 数据集中 | 迁移/发布风险高，破坏回滚边界 | 不选 |
| 独立 SQLite sidecar + 精确向量扫描 | 最少运维，易打包和删除 | 大规模时查询变慢 | MVP 推荐 |
| 本地 ANN 库 | 大规模查询快 | 原生依赖、打包、增量一致性更复杂 | 实测后再选 |
| Qdrant 等独立服务 | 检索能力完整 | 对桌面离线工具过重 | V1 不选 |
| 显式 Python RAG 链 | 可读、可测试、依赖少 | 需要自己维护少量编排代码 | 推荐 |
| LangChain/LangGraph | 生态集成多 | 增加抽象和调试成本 | V1 暂不引入 |
| 直接混合原始分数 | 实现短 | 分数尺度不一致 | 不选 |
| RRF 融合 | 稳定、可解释、无标定要求 | 不利用绝对分数 | V1 推荐 |

## 14. 需要用户确认的 5 个决策

1. **数据出网边界**：只允许完全本地，还是允许把公开测试资料片段发给指定外部 LLM？默认建议完全本地/Mock。
2. **运行硬件**：目标演示电脑的 CPU、内存、是否有可用 GPU；它决定本地 Embedding/LLM 的体积和延迟目标。
3. **首批测试语料**：确认只使用公开许可资料或人工构造资料，并确定 PDF/DOCX/TXT 的代表性样本。
4. **V1 交付形态**：先做命令行/测试可验证的核心链路，再接现有 UI，还是第一周就要 UI；建议先核心后 UI。
5. **安装体积和依赖容忍度**：能否接受增加本地模型与推理运行时；若不能，需选择经批准的在线 provider。

## 15. 暂时不做

- Agent、多 Agent、自动规划：V1 先证明检索与引用可信。
- 全格式一次接入：先用 PDF/DOCX/TXT 建立评测基线。
- 自动“学习”企业文档或长期记忆：权限、遗忘和审计尚未设计。
- 云端向量数据库、微服务、Kubernetes：与单机桌面和每周 5～7 小时约束不匹配。
- 复杂 reranker：先用检索指标证明是否需要。
- 自动宣称质量提升：所有 Recall@K、MRR、忠实度和延迟数字必须来自同一标注集的实际运行。

## 16. V1 之前必须先完成的基线修复

在开始真实 embedding/OCR 语料前，先解决 `ocr_models/manifest.json` 与模型文件不一致，并复跑相关 OCR、PDF、ZIP 测试。修复方式必须是恢复正确模型文件或在确认模型包完整后重新生成 manifest，不能绕过 `model_manifest_fingerprint()` 校验。
