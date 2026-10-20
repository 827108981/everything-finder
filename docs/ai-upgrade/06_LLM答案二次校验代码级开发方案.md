# 专业版 RAG 全链路修正与 LLM 二次校验代码级开发方案

> 本文替代原来的“只增加第二次 LLM 校验”方案。二次校验只能检查已经进入上下文的证据，无法挽救在召回、去重、重排或 TopK 截断时丢失的正确证据。因此本方案修正完整链路，而不是只在末尾增加一次 API 调用。

## 1. 总体结论

### 1.1 保留的现有组件

- `RagIndexStore + ChromaVectorStore`：保存块元数据与向量索引。
- `LexicalRetriever + VectorRetriever`：BM25/FTS 与语义召回互补。
- `HybridRetriever`：RRF 融合不同量纲的排名。
- `LocalBgeReranker`：Cross-Encoder 精排少量候选。
- `ContextBuilder`：控制上下文长度并生成 `[S1]` 引用。
- `OpenAICompatibleGateway`：隔离 Responses 与 Chat Completions。
- `RagQueryWorker + QThread`：避免模型和网络调用卡住 Qt 主线程。
- AI 索引独立发布：新索引未发布时继续查询旧索引。

### 1.2 当前缺口

1. `QueryAnalyzer` 只有字符串规则，没有结构化表达“单项目/全部项目”等条件。
2. 固定四字窗口会产生“项目的发”“时候该怎”等低价值检索词。
3. `ContextBuilder` 才去重，重复块已抢占重排和最终 TopK。
4. 通用 BGE 判断主题相关性，不保证所有业务条件同时满足。
5. `EvidencePolicy` 只有两个分数阈值，没有条件冲突和证据覆盖判断。
6. 缺少必要信息时没有澄清分支。
7. 生成后只检查引用编号存在，不检查结论是否真被正文支持。

### 1.3 目标链路

```text
问题
 -> QueryInterpreter：结构化意图 + BM25 词
 -> ClarificationPolicy：必要时追问
 -> BM25/FTS + 向量召回
 -> RRF 融合
 -> CandidateDeduplicator：重排前去重
 -> BGE Cross-Encoder 重排（暂不截 final_top_k）
 -> ConstraintMatcher：业务条件一致性
 -> EvidencePolicy：相关性 + 条件覆盖门禁
 -> ContextBuilder
 -> LLM 第一次生成初稿
 -> Python 引用预检查
 -> LLM 第二次审查并修订
 -> Python 最终硬校验
 -> 展示与完整诊断
```

职责口诀：召回负责“不漏”，重排负责“更像”，约束层负责“场景对不对”，二次 LLM 负责“答案有没有证据”，Python 负责“格式与安全底线”。

## 2. 文件级改造清单

```text
local_full_text_search/rag/
  models.py                       修改：QueryPlan、约束、审查结果
  query_analyzer.py               重构：确定性词项与条件抽取
  query_interpreter.py            新增：生成统一 QueryPlan
  clarification.py                新增：检索前/后的澄清判断
  candidate_deduplicator.py       新增：重排前精确/近重复去除
  constraint_matcher.py           新增：业务条件一致性检查
  evidence_policy.py              重构：联合证据门禁
  context_builder.py              修改：接受已决策候选
  answer_review.py                新增：严格解析二次审查 JSON
  citation_validator.py           修改：预检查 + 最终硬校验
  answer_service.py               重构：完整编排
  trace.py                        修改：新增诊断字段
  generation/base.py              修改：生成/审查请求协议
  generation/openai_compatible.py 修改：两类 Prompt、公共请求方法
  retrieval/base.py               修改：传入 QueryPlan
  retrieval/lexical.py            修改：不再自行重复分析
  retrieval/vector.py             修改：使用 semantic_query
  retrieval/hybrid.py             修改：共享 QueryPlan

local_full_text_search/workers/rag_query_worker.py
local_full_text_search/ui/rag_page.py
local_full_text_search/ui/rag_diagnostic_dialog.py
local_full_text_search/config/defaults.py
```

## 3. 数据模型

修改 `local_full_text_search/rag/models.py`。

### 3.1 查询计划与约束

```python
from enum import StrEnum

class ConstraintKind(StrEnum):
    SCOPE = "scope"          # single / multiple / all
    STAGE = "stage"          # calibration / qc / sample / startup
    OBJECT = "object"        # 发光值、底物、试剂、项目
    DIRECTION = "direction"  # high / low / abnormal / failed
    MODEL = "model"

@dataclass(frozen=True, slots=True)
class QueryConstraint:
    kind: ConstraintKind
    value: str
    required: bool = True
    source_text: str = ""
    confidence: float = 1.0

@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_text: str
    normalized_text: str
    intent: str
    entities: tuple[str, ...]
    constraints: tuple[QueryConstraint, ...]
    lexical_terms: tuple[str, ...]
    lexical_expression: str
    semantic_query: str
    missing_fields: tuple[str, ...] = ()
    analyzer_version: str = "query-plan-v1"
```

- `lexical_terms` 给 BM25/FTS，必须是正文中可能出现的词。
- `semantic_query` 给 BGE embedding，默认保留完整原问题。
- `constraints` 给候选条件检查。
- `missing_fields` 给澄清机制。

### 3.2 候选判断

```python
class ConstraintStatus(StrEnum):
    MATCH = "match"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class ConstraintJudgement:
    constraint: QueryConstraint
    status: ConstraintStatus
    reason: str
    matched_text: str = ""

@dataclass(frozen=True, slots=True)
class CandidateDecision:
    stable_chunk_id: str
    accepted: bool
    reason: str
    judgements: tuple[ConstraintJudgement, ...]
    required_match_count: int
    required_unknown_count: int
    required_conflict_count: int
```

向 `EvidenceChunk` 添加：

```python
constraint_score: float | None = None
constraint_decision: CandidateDecision | None = None
duplicate_of: str | None = None
```

不要把 `constraint_score` 与 `rerank_score` 粗暴相加；一个表示场景一致性，一个表示主题相关性，诊断时必须分开看。

### 3.3 回答状态

```python
class RagAnswerStatus(StrEnum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    INSUFFICIENT = "insufficient"

@dataclass(frozen=True, slots=True)
class RagAnswer:
    # 保留现有字段
    status: RagAnswerStatus = RagAnswerStatus.ANSWERED
    clarification_question: str = ""
    audit: AnswerAudit | None = None
```

暂时保留 `insufficient_evidence`，兼容现有 UI 和测试。

## 4. QueryAnalyzer：只做确定性抽取

重构 `rag/query_analyzer.py`：

```python
@dataclass(frozen=True, slots=True)
class DeterministicAnalysis:
    normalized_text: str
    identifiers: tuple[str, ...]
    domain_terms: tuple[str, ...]
    status_terms: tuple[str, ...]
    synonym_groups: tuple[tuple[str, ...], ...]
    constraints: tuple[QueryConstraint, ...]

class QueryAnalyzer:
    def analyze(self, text: str) -> DeterministicAnalysis: ...
```

删除当前 `_search_terms()` 的任意中文固定窗口切片。只做：

1. 型号、数值、英文缩写识别；
2. 稳定业务术语识别；
3. 同义词扩展；
4. 去掉“怎么、如何、请问”等提问动作词；
5. 明确字面条件抽取。

新增 `rag/resources/query_lexicon.json`：

```json
{
  "terms": ["发光值", "底物", "定标", "质控", "试剂", "项目"],
  "synonyms": {
    "未通过": ["不通过", "失败"],
    "偏低": ["低", "下降", "不足"],
    "异常": ["故障", "报错", "不正常"]
  },
  "constraints": {
    "scope.single": ["单个项目", "某个项目", "只有一个项目", "仅一个项目"],
    "scope.multiple": ["多个项目", "部分项目"],
    "scope.all": ["全部项目", "所有项目", "所有检测项目"]
  }
}
```

词典是稳定、可测试的业务表达，不是完整自然语言理解器。未知表达仍由向量通道覆盖。

## 5. QueryInterpreter：生成一次、全链路共用

新增 `rag/query_interpreter.py`：

```python
class QueryInterpreter:
    def __init__(self, analyzer, llm_parser=None): ...

    def interpret(self, text, token) -> QueryPlan:
        deterministic = self.analyzer.analyze(text)
        llm_result = self.llm_parser.parse(text, token) if self.llm_parser else None
        return self._merge(text, deterministic, llm_result)
```

默认 `llm_parser=None`，不增加网络调用。专业模式以后可启用严格 JSON 的 LLM 分析，但规则与 LLM 冲突时，以明确字面规则为准；LLM 失败时回退规则。

BM25 表达式由词项构造：同义词组内部 `OR`，不要先让 LLM 把问题改写成新句子再整句送 FTS。完整原问句始终作为 `semantic_query`。

## 6. 澄清机制

新增 `rag/clarification.py`：

```python
@dataclass(frozen=True, slots=True)
class ClarificationDecision:
    required: bool
    question: str
    missing_fields: tuple[str, ...]
    reason: str

class ClarificationPolicy:
    def evaluate(self, plan: QueryPlan) -> ClarificationDecision: ...
```

只有缺失条件会导致完全不同的处理路径时才追问：

- “发光值异常怎么排查”可能需确认定标、质控还是样本检测。
- “只有某个项目发光值异常”可先检索；若高分证据分成互斥阶段，再动态追问。
- “CL-8000i 定标时只有一个项目发光值偏低”信息充分，不追问。

第一版不保存多轮会话。返回 `NEEDS_CLARIFICATION`，用户补充后重新提问；不要把澄清当错误弹窗。

## 7. 召回层共享 QueryPlan

修改 `retrieval/base.py`：

```python
class Retriever(Protocol):
    def retrieve(self, query: RagQuery, plan: QueryPlan, token) -> list[EvidenceChunk]: ...
```

修改 `retrieval/lexical.py`：删除内部 `QueryAnalyzer()`，直接使用 `plan.lexical_expression/terms`，避免同一问题分析两次得到不一致结果。

修改 `retrieval/vector.py`：

```python
query_vector = self.embedding.embed_query(plan.semantic_query)
```

修改 `retrieval/hybrid.py`：

```python
lexical = self.lexical.retrieve(query, plan, token)
vector = self.vector.retrieve(query, plan, token)
return self._rrf_fuse(lexical, vector, query.fusion_top_k * 2)
```

RRF 保持 `1 / (rrf_k + rank)`。它用于融合 BM25 与余弦相似度的排名，不负责判断业务条件。

## 8. 重排前去重

新增 `rag/candidate_deduplicator.py`，位置必须在 RRF 后、BGE 前：

```python
@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    removed_chunk_id: str
    kept_chunk_id: str
    method: str
    similarity: float

@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    candidates: tuple[EvidenceChunk, ...]
    duplicates: tuple[DuplicateRecord, ...]

class CandidateDeduplicator:
    def deduplicate(self, candidates, *, limit) -> DeduplicationResult: ...
```

规则顺序：

1. `stable_chunk_id` 相同；
2. `text_hash` 相同；
3. 同文档规范化正文字符 3-gram Jaccard `>= 0.92`。

第一版不跨文档模糊去重，不同文件的相同说明可能代表不同版本。RRF 可先取 `fusion_top_k * 2`，去重后截到 `fusion_top_k`，避免去重后候选不足。

## 9. BGE 重排职责调整

保留 `rag/reranking/local_bge.py` 的 Cross-Encoder 推理，但调整调用方式：

```python
scored = reranker.rerank(
    plan.original_text,
    deduped.candidates[:query.reranker_top_k],
    top_n=query.reranker_top_k,
    token=token,
)
```

关键变化：

- 输入完整原问题，不用 BM25 表达式替代。
- 暂不截 `final_top_k`，否则正确证据在条件检查前已丢失。
- 把 `_apply_document_diversity()` 从重排器移到最终证据选取。
- 重排器只打分排序，不承担业务规则。

## 10. ConstraintMatcher：修复本次核心框架问题

新增 `rag/constraint_matcher.py`：

```python
class ConstraintMatcher:
    def evaluate(self, plan, candidates) -> tuple[EvidenceChunk, ...]: ...
```

例如范围词：

```python
SCOPE_PATTERNS = {
    "single": ("单个项目", "某个项目", "只有一个项目", "仅一个项目"),
    "multiple": ("多个项目", "部分项目"),
    "all": ("全部项目", "所有项目", "所有检测项目", "大面积"),
}
```

判断顺序：

1. 候选明确出现同值表达 -> `MATCH`；
2. 明确出现互斥值 -> `CONFLICT`；
3. 没描述该条件 -> `UNKNOWN`，不能误判冲突。

否定短语必须优先处理，“不是所有项目”不能命中 `all`。任一 required 条件明确冲突，候选不能成为主要证据；全部 UNKNOWN 可作补充证据，但不能仅凭高 BGE 分成为第一证据。

后续可给 Top 10 增加一次批量 LLM Evidence Judge，只处理规则的 `UNKNOWN`；不得推翻明确字面冲突。首版默认关闭，避免一次问答变成 3～4 次 API 调用。

## 11. EvidencePolicy 联合门禁

重构 `rag/evidence_policy.py`：

```python
class EvidencePolicy:
    def evaluate(self, query, plan, candidates) -> EvidenceDecision: ...
```

`EvidenceDecision` 增加 `rejected`、`needs_clarification`、`clarification_question`。

单条候选检查顺序：

1. `rerank_score` 最低阈值；
2. 命中关键词或向量分达到最低值；
3. required constraint 无明确冲突；
4. 不是已记录重复项；
5. 同文档块数未超限。

集合级检查：

1. 至少一条无冲突证据；
2. required 条件至少被一条证据明确覆盖；
3. 高分证据是否存在互斥场景，必要时转澄清；
4. 最后才按 `final_top_k`、`max_per_document=3` 选取。

不要把 BGE、向量和约束分直接相加。采用“硬门禁后按 rerank 排序”，更可解释、可测试。

## 12. ContextBuilder

修改 `rag/context_builder.py`。保留 `text_hash` 去重作为防御性最后一道，但主要去重已经提前。

上下文格式：

```text
[S1]
文件：xxx.pdf
位置：第 41 张幻灯片
系统检索判断：与问题范围“单个项目”一致
内容：原始正文……
```

必须标明“系统检索判断”，不能伪装成原文。`ContextBundle` 增加 `evidence_ids` 与 `context_hash`，常规日志只记 hash 和长度，不记正文。

## 13. 第一次 LLM：生成初稿

修改 `generation/base.py`，给 `GenerationRequest` 添加 `query_plan_summary`。生成 Prompt 增加：

```text
必须保留问题中的范围、阶段、对象、方向。
不得把“全部/多个项目异常”的步骤直接用于“单项目异常”，除非证据明确说明相同。
每条事实性结论必须紧邻 [S#]。
证据不足只输出“资料中没有足够依据。”。
```

第一次输出自然语言，不强制 JSON，保证答案可读。

## 14. 引用预检查与最终硬校验

修改 `rag/citation_validator.py`：

```python
@dataclass(frozen=True, slots=True)
class CitationCheck:
    cited_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    has_citations: bool
    structurally_valid: bool
    reason: str

def inspect(self, answer_text, source_map) -> CitationCheck: ...
```

第一次答案无引用或引用不存在时，不展示初稿，把结构问题交给第二次 LLM 修订。第二次之后仍调用严格 `validate()`：未知引用、非资料不足答案无引用均失败。

## 15. 第二次 LLM：审查并直接返回可发布答案

### 15.1 协议

修改 `rag/generation/base.py`：

```python
@dataclass(frozen=True, slots=True)
class AnswerReviewRequest:
    question: str
    query_plan_summary: str
    context: str
    draft_answer: str
    citation_check_summary: str
    model: str
    api_base_url: str
    api_style: str
    api_key: str
    timeout_seconds: int = 90
    max_output_tokens: int = 1400

@dataclass(frozen=True, slots=True)
class ClaimReview:
    claim: str
    citations: tuple[str, ...]
    status: str  # supported / contradicted / not_supported
    reason: str

@dataclass(frozen=True, slots=True)
class AnswerReviewResult:
    verdict: str  # pass / revised / insufficient
    summary: str
    claims: tuple[ClaimReview, ...]
    final_answer: str
    provider_name: str
    model_name: str

class LLMGateway(Protocol):
    def generate(self, request, token) -> ModelAnswer: ...
    def review_answer(self, request, token) -> AnswerReviewResult: ...
```

### 15.2 严格 JSON

```json
{
  "verdict": "revised",
  "summary": "删除了一条没有证据支持的步骤",
  "claims": [{
    "claim": "先检查该项目对应的试剂和参数",
    "citations": ["S1"],
    "status": "supported",
    "reason": "S1 明确给出单项目异常的排查路径"
  }],
  "final_answer": "先检查……[S1]"
}
```

- `pass`：可保持初稿，但仍返回完整 `final_answer`。
- `revised`：删除或改写不受支持内容。
- `insufficient`：最终答案必须严格为“资料中没有足够依据。”。
- unsupported/contradicted 结论不得继续出现在 `final_answer`。

新增 `rag/answer_review.py` 的 `AnswerReviewParser`：只允许一个 JSON 对象；最多剥一层 JSON 围栏；校验枚举、字段长度、claims 数量和 S 编号白名单；拒绝 JSON 前后解释文字。

### 15.3 网关重构

修改 `generation/openai_compatible.py`：

```python
def provider_generation_payload(request): ...
def provider_review_payload(request): ...

class OpenAICompatibleGateway:
    def generate(...):
        text = self._request_text(provider_generation_payload(...), ...)

    def review_answer(...):
        text = self._request_text(provider_review_payload(...), ...)
        return parser.parse(text, allowed_source_ids=...)

    def _request_text(self, payload, request, token) -> str:
        # 统一 SDK、Responses/Chat、异常翻译和 close
```

不要复制两套 401/404/429/超时处理。

### 15.4 失败策略

默认严格：第二次网络失败或 JSON 非法时，不展示未经审查的初稿；提示“答案审查失败，请重试”。关闭二次审查时允许一次调用，但回答必须带“未启用语义审查”警告。

正常路径只需两次 LLM：第一次生成，第二次审查并直接修订，不需要第三次生成。

## 16. AnswerService 完整编排

重构 `rag/answer_service.py`，新增可注入组件，测试可传 Fake：

```python
query_interpreter=None
clarification_policy=None
deduplicator=None
constraint_matcher=None
evidence_policy=None
```

目标伪代码：

```python
def answer(self, query, token, *, trace_id=None):
    plan = self.query_interpreter.interpret(query.text, token)
    self._report("query_interpreted", query_plan=safe_snapshot(plan))

    clarification = self.clarification_policy.evaluate(plan)
    if clarification.required:
        return self._clarification(trace_id, clarification)

    with self.source_generation_store.acquire_active() as (source, lease):
        bundle = self._open_compatible_rag_bundle(source)
        vector_store = ChromaVectorStore(bundle.chroma_path)
        try:
            fused = hybrid.retrieve(query, plan, token)
            if not fused:
                return self._insufficient(trace_id, "no_candidates")

            deduped = self.deduplicator.deduplicate(
                fused, limit=query.fusion_top_k
            )
            scored = self.reranker.rerank(
                plan.original_text,
                deduped.candidates[:query.reranker_top_k],
                query.reranker_top_k,
                token,
            )
            constrained = self.constraint_matcher.evaluate(plan, scored)
            decision = self.evidence_policy.evaluate(query, plan, constrained)
            if decision.needs_clarification:
                return self._clarification(trace_id, decision)
            if decision.insufficient:
                return self._insufficient(trace_id, decision.reason)

            context = self.context_builder.build(decision.accepted, ...)
            credentials = self._load_credentials()
            draft = self.gateway.generate(
                self._generation_request(plan, context, credentials), token
            )
            citation_check = self.citation_validator.inspect(
                draft.text, context.source_map
            )

            if self.settings.rag_answer_review_enabled:
                review = self.gateway.review_answer(
                    self._review_request(
                        plan, context, draft, citation_check, credentials
                    ), token
                )
                final_text = review.final_answer
            else:
                review = AnswerReviewResult.disabled(draft.text)
                final_text = draft.text

            if final_text == "资料中没有足够依据。":
                return self._reviewed_insufficient(...)

            cited_ids = self.citation_validator.validate(
                final_text, context.source_map
            )
            return RagAnswer(
                answer_text=final_text,
                sources=tuple(context.source_map[x] for x in cited_ids),
                source_ids=cited_ids,
                audit=self._build_audit(plan, draft, review, citation_check),
                ...,
            )
        finally:
            vector_store.close()
```

第二次 LLM 必须看到原问题、结构化约束、最终证据和初稿；它不扫描 50 条候选，也不替代 BGE。

## 17. Worker、取消与日志

修改 `workers/rag_query_worker.py`，新增阶段：

```text
query_interpreting -> clarification_check -> lexical_retrieval
-> vector_retrieval -> fusion -> deduplicating -> reranking
-> checking_constraints -> checking_evidence -> building_context
-> draft_generating -> draft_citation_check -> answer_reviewing
-> final_validating -> completed
```

每个循环和网络调用前后检查 `CancelToken`。若 SDK 阻塞调用无法立即中断，请求返回后必须丢弃结果并发出 `cancelled`，不能显示过期答案。

日志只记 request id、阶段、耗时、候选/冲突/去重数量、模型、接口类型、host、review verdict、错误类型与 HTTP 状态。禁止记录 Key、完整问题、正文、初稿、最终答案和完整 SDK 请求对象。

## 18. 诊断界面

修改 `ui/rag_diagnostic_dialog.py`，页签顺序：

```text
问题理解 | 关键词召回 | 向量召回 | RRF融合 | 候选去重
| BGE重排 | 条件一致性 | 证据门禁 | 引用上下文
| 答案初稿 | LLM二次校验 | 最终结果
```

“条件一致性”至少显示：

```text
重排名次 | 文件 | 位置 | BGE分数 | 范围 | 阶段 | 方向 | 结论 | 原因
```

本案例应能显示：

```text
候选A | 0.981 | 范围=CONFLICT | 淘汰 | 正文描述几乎所有项目失败
候选B | 0.782 | 范围=MATCH    | 保留 | 正文描述单项目异常
```

二次校验页显示 verdict、摘要、每条 claim/引用/状态/理由以及初稿是否被修改。保留现有一键复制诊断文本和全部页签截图，增加脱敏 JSON 导出；默认不导出正文和用户本地完整路径。

`rag_page.py` 对 `NEEDS_CLARIFICATION` 展示追问、清空来源，不弹失败框。后续可增加“定标/质控/样本检测”快捷按钮。

## 19. 配置与迁移

`config/defaults.py` 配置版本加 1，新增：

```python
rag_query_interpreter_mode: str = "deterministic"
rag_clarification_enabled: bool = True
rag_dedup_jaccard_threshold: float = 0.92
rag_max_chunks_per_document: int = 3
rag_constraint_matching_enabled: bool = True
rag_llm_evidence_judge_enabled: bool = False
rag_answer_review_enabled: bool = True
rag_answer_review_max_output_tokens: int = 1400
rag_answer_review_strict: bool = True
```

新安装默认开确定性分析、去重、约束检查和二次审查。旧安装升级时前三项可直接开；二次审查先关闭并提示“开启后每次问答通常调用 LLM 两次”。LLM Query Interpreter 和 Evidence Judge 默认关闭。

设置页显示预计调用次数：生成一次 + 审查一次。

## 20. 测试与 Gold 集

### 20.1 新增/扩展测试文件

- `test_query_analyzer.py`：不产生无意义四字窗口；正确抽取 single、型号、方向；处理否定。
- `test_query_interpreter.py`：规则结果、可选 LLM 合并、冲突规则优先、非法 JSON 回退。
- `test_clarification.py`：该追问/不该追问、检索后冲突追问。
- `test_candidate_deduplicator.py`：ID、hash、同文档近重复、跨文档保留。
- `test_constraint_matcher.py`：single 对 all 为冲突，未描述为 UNKNOWN，否定不误判。
- `test_evidence_policy.py`：高 BGE 冲突项淘汰；较低 BGE 匹配项晋级；正确证据在第 9 也能进入最终 8。
- `test_answer_review.py`：三种 verdict、未知 S、非法 JSON、unsupported 泄漏到 final 的拒绝。
- `test_generation_gateway.py`：两种 API、公共错误映射、无 Key 泄漏。
- `test_rag_end_to_end.py`：正常两次调用、审查修订、审查不足、严格失败、取消、关闭审查。

### 20.2 业务 Gold 集

新增 `tests/rag/fixtures/gold_cases.jsonl`：

```json
{
  "id": "single-project-luminescence-abnormal",
  "question": "只有某个项目的发光值异常，这个时候该怎么排查",
  "required_constraints": {"scope": "single", "object": "发光值"},
  "positive_chunk_ids": ["正确证据块ID"],
  "negative_chunk_ids": ["全部项目异常块ID", "多项目异常块ID"],
  "must_retrieve_at_k": 30,
  "must_survive_final_k": 8
}
```

指标：`Recall@30`、MRR、`ConstraintViolation@8`、`EvidencePrecision@8`、引用合法率、Claim 支持率、澄清准确率。以后换模型、词典或阈值都跑固定集，不能只凭一条答案感觉好坏。

## 21. 分阶段实施

### A. 先修候选质量

QueryPlan -> Retriever 共享计划 -> 重排前去重 -> 延后 final TopK -> ConstraintMatcher -> EvidencePolicy -> Gold 测试。

验收门：本次“单项目”案例中，“全部/多项目异常”不能挤掉正确单项目证据。

### B. 澄清机制

ClarificationPolicy -> RagAnswer 新状态 -> UI 展示 -> 澄清测试集。

### C. 二次 LLM 发布审查

审查模型/解析器 -> 网关公共请求 -> AnswerService 接入 -> 严格失败策略。

### D. 诊断和评估闭环

新增页签 -> 脱敏导出 -> Gold 批量评估 -> 比较每次算法升级指标。

不要四阶段一次混合开发；每阶段通过测试和 Gold 门禁后再进入下一阶段，否则无法定位偏差来自召回、条件判断还是 LLM。

## 22. 验收标准

- 新全文索引未发布时，旧 AI 索引继续可用；重启后读取已发布 Chroma，不重复下载/重建。
- 检索词不含无意义四字窗口。
- 重复候选在 BGE 前去除并可诊断。
- required 条件冲突项不能仅凭高 BGE 分进入主要证据。
- 缺失关键条件时澄清，不猜测。
- 正常答案调用两次 LLM，第二次可修订；失败时不泄漏初稿。
- 最终所有事实结论有合法 S 引用。
- Gold 集 `Recall@30 >= 0.95`。
- 已覆盖规则的 `ConstraintViolation@8 == 0`。
- `CitationValidity == 1.0`。
- 记录各阶段 P50/P95；取消查询不会让旧答案覆盖新查询。
- Key 不进入日志、诊断、导出和异常文本。

## 23. 本次问题改造后的实际走向

对于“只有某个项目的发光值异常，这个时候该怎么排查”：

1. QueryPlan 抽取 `scope=single`、`object=发光值`、`intent=troubleshooting`。
2. BM25 检索稳定术语，向量检索保留完整问题。
3. RRF 汇合两条通道；重排前删除重复页/幻灯片。
4. BGE 仍可能把“所有项目异常”排高，这不算模型故障，它只判断主题相关。
5. ConstraintMatcher 将“全部/多项目异常”标为 `CONFLICT`。
6. EvidencePolicy 淘汰冲突项，使原 BGE 第 7 的正确“单项目异常”证据晋级。
7. 第一次 LLM 只根据无冲突证据生成初稿。
8. 第二次 LLM 逐条验证排查步骤，删除越界推断并返回最终稿。
9. Python 确认所有 S 引用存在后才展示。

因此，二次 LLM 是最后一道发布门禁，不是替代查询理解、候选治理和证据决策的万能补丁。
