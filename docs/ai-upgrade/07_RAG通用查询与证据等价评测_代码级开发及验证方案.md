# RAG 通用查询、证据等价与可追溯评测：代码级开发及验证方案

> 状态：已完成第二轮代码对照审查，待开发实施<br>
> 文档修订：v1.1（2026-10-26）；本次仅修订方案，未实施业务改造<br>
> 编写日期：2026-10-26<br>
> 适用仓库：`E:\code\everything-finder`<br>
> 基线报告：`C:\Users\50263194\AppData\Local\LocalFullTextSearch\rag_eval_runs\20261026_113214_421f57`<br>
> 基线索引：`g000008-9f01fc283b`

## 1. 文档目的

本方案解决当前 8 道生化/免疫对照题暴露出的四类问题：

1. 查询解析依赖固定领域词和固定“范围/阶段/方向”结构，导致 `P02` 被当作设备型号、“样本针”被当作样本检测阶段。
2. 关键词查询包含问句碎片，所有词以同一种方式进入 FTS `OR` 查询，无法突出型号、部件编号和完整对象描述的共同出现。
3. PDF、PPTX、DOCX 等不同来源中的排版等价正文未统一计为证据；相反，同文档近似正文可能仅凭相似度被删除。
4. 批量评测、离线 HTML 和阶段诊断使用了不同的证据匹配口径，报告无法稳定回答“金标在哪里消失、为什么消失”。

本方案只使用问题自身和索引正文中的信息。Gold 数据只用于离线评分和报告标色，不能参与线上查询生成、召回、融合、重排或证据门禁。

文中代码为目标接口及核心逻辑示意，不表示功能已经存在。新增类型统一使用 `from __future__ import annotations`，实现时补齐导入、序列化和校验；示意中的 `...` 不是可交付实现。v1.1 的审查结论见第 18 节。

## 2. 已核实的当前实现

### 2.1 查询与检索

- `rag/query_analyzer.py` 使用 `_DOMAIN_TERMS`、`_SCOPE_PATTERNS`、`_STAGE_PATTERNS`、`_DIRECTION_PATTERNS` 生成词项和条件。
- 所有字母数字组合标识符都会生成 `ConstraintKind.MODEL`；因此 `BS-2800` 与 `P02` 可能成为两个互相冲突的“型号”。
- `_STAGE_PATTERNS["sample_test"]` 包含裸词“样本”和“标本”，会把“样本针”误判为检测阶段。
- 中文剩余文本被切成 3/4 字窗口，可能出现“考光路有”“有什么作”等问句碎片。
- `rag/index_store.py::lexical_search()` 对全部词项构造一个 FTS5 trigram `OR` 表达式，未保存每条查询分支的贡献。
- `rag/retrieval/vector.py` 只嵌入完整原问题。
- `rag/retrieval/hybrid.py` 用 RRF 合并关键词和向量两条路线。

### 2.2 去重与证据匹配

- `rag/content_identity.py` 已有正文规范化入口，应在此扩展有版本的公共实现。但现实现并不保证事实保真：大小写折叠、NFKC 和删除空白/符号可能丢失单位、指数、表格边界和方向信息，必须补充保护。
- `rag/candidate_deduplicator.py` 已能跨文件合并 `canonical_content()` 完全相同的正文并保留多个来源。
- 同文档候选仍可能因为 3-gram Jaccard `>= 0.92` 被直接删除；这可能丢失仅出现在重叠块尾部的关键事实。
- `rag/evaluation/evidence_matcher.py` 在运行前按“同文件、同页、引文”把 Gold 映射到索引 chunk。
- `rag/evaluation/reporting/matching.py` 生成 HTML 时再次按“文件名、页码、引文 LCS”匹配。
- 两套匹配器可能产生不同结论，导致批量指标与 HTML 标色不一致。

### 2.3 当前基线

基线运行记录的模式为 `retrieval-only`，模型配置为 BGE Embedding/Reranker，API 请求数为 0。以下数字读取自保存的 `summary.md`，本轮未重跑；新验证必须另行记录实际加载的模型和后端，不能只凭配置字段声称真实模型测试通过：

| 阶段 | Gold 命中题数 | 平均证据召回率 | MRR |
|---|---:|---:|---:|
| 关键词召回 | 3/8 | 25.0% | 15.9% |
| 向量召回 | 3/8 | 37.5% | 2.5% |
| 融合排序 | 5/8 | 50.0% | 6.3% |
| 去重 | 4/8 | 37.5% | 6.1% |
| 重排 | 2/8 | 18.8% | 7.3% |
| 最终上下文 | 2/8 | 18.8% | 9.4% |

当前报告已确认的典型现象：

- 004 的指定 PDF 金标未命中，但内容对应的 PPTX 在关键词第 2、最终上下文第 1；当前评分仍可能把它算作漏召回。
- 007 的两条金标只命中“位置/归属”依据，没有命中“作用”依据；单独显示 `1/2` 无法说明缺失事实。
- 001、005、008 的金标曾进入前 50，但在融合或重排入口截断。
- 003、006 的指定 Gold 在关键词和向量前 50 中均未出现。
- 单题主要耗时在 Cross-Encoder 重排，扩大候选数会显著增加总耗时，不能直接把 TopK 全部调大。

## 3. 设计原则与范围

### 3.1 查询解析采用“原文保真 + 开放特征”

查询解析首先提取原文中可直接观察的信息：原文片段、字符位置、标识符形态、数字单位、连续短语和疑问意图。它不要求每个对象必须属于预先规定的“设备/模块/子模块”类别。

正文用于：

- 计算词项在语料中的区分度；
- 说明候选是否支持某个明确条件；
- 在报告中解释角色或适用范围；

正文不会反向篡改用户原问题。P0 不强制执行第二轮检索；第二轮只能作为有开关、有独立指标的后续实验。

### 3.2 “范围、阶段、方向”降为可选属性

这些字段继续保留以读取旧结果，但不再作为所有问题必须具备的通用结构。规则解析得到的范围、阶段、方向默认只能软排序或显示，不能因为正文未提及就删除候选。

只有同时满足以下条件时才能硬拒绝：

1. 条件由用户明确写出，而不是从包含关系或词表推断；
2. 候选在与目标事实相同的局部句、表格行或明确标题中表达互斥条件；
3. 诊断记录能够展示问题条件、候选冲突文本和判定原因。

候选没有写明型号或条件时记为 `unknown`，继续保留在候选池；这不意味着已经确认适用于目标设备。最终生成必须区分已确认事实与不确定内容，详见 8.3。

### 3.3 关键词没有固定“三类”

解析器输出开放的检索信号。一个信号可同时具有多个特征，例如：

```json
{
  "text": "P02",
  "start": 15,
  "end": 18,
  "features": ["ascii_digit_identifier", "exact_surface"],
  "source": "question"
}
```

`features` 是事实记录，不是固定业务分类，也不直接等于分值。`high_idf` 不能由解析器凭空写入；只有检索层实际查询语料统计后，才能在路线诊断中记录 `df/N/统计口径`。P0 不依赖 IDF 生成查询。

### 3.4 95% 相似只产生“疑似”，自动等价必须解释差异

- 去除排版噪声后事实字符完全一致：自动判为排版等价。
- 规范化相似度高，但仍存在文字差异：标为疑似等价，默认不计分、不合并。
- 数字、单位、型号、编号、范围符号、比较符号或否定含义不同：标为事实冲突，禁止合并。

这正对应“95% 正文相同，剩余 5% 只是格式/标点时视为同一内容；剩余 1% 含实质内容时不能视为同一内容”。百分比用于寻找候选，差异分类决定能否自动确认。

## 4. 目标数据模型

### 4.1 查询表示

修改 `rag/models.py`，新增以下兼容结构：

```python
@dataclass(frozen=True, slots=True)
class QuerySpan:
    text: str
    normalized_text: str
    start: int             # 原问题字符下标，左闭右开
    end: int
    features: tuple[str, ...]
    source: str = "question"


@dataclass(frozen=True, slots=True)
class QueryTransformation:
    operation: str          # trim_prefix / trim_suffix / remove_question_shell
    source_text: str
    result_text: str
    reason: str


@dataclass(frozen=True, slots=True)
class RetrievalQueryVariant:
    route_id: str           # lexical_precise / lexical_phrase / lexical_broad
    query_text: str
    expression: str = ""
    signal_texts: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalRouteHit:
    route_id: str
    rank: int
    score: float | None
    score_kind: str = "unknown"  # bm25_raw / vector_similarity；禁止填 RRF
```

扩展现有 `QueryPlan`：

```python
@dataclass(frozen=True, slots=True)
class QueryPlan:
    # 保留现有字段，避免调用方同时迁移失败
    original_text: str
    normalized_text: str
    intent: str
    entities: tuple[str, ...]
    constraints: tuple[QueryConstraint, ...]
    lexical_terms: tuple[str, ...]
    lexical_expression: str
    semantic_query: str
    fallback_terms: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    analyzer_version: str = "query-plan-v3"  # 保持原字段位置

    # 新字段
    spans: tuple[QuerySpan, ...] = ()
    transformations: tuple[QueryTransformation, ...] = ()
    lexical_variants: tuple[RetrievalQueryVariant, ...] = ()
    semantic_variants: tuple[RetrievalQueryVariant, ...] = ()
    @property
    def terms(self) -> tuple[str, ...]:
        return self.lexical_terms

    @property
    def fts_expression(self) -> str:
        return self.lexical_expression
```

扩展 `EvidenceChunk`：

```python
lexical_route_hits: tuple[RetrievalRouteHit, ...] = ()
vector_route_hits: tuple[RetrievalRouteHit, ...] = ()
lexical_route_rrf_score: float | None = None
vector_route_rrf_score: float | None = None
```

现有 `lexical_rank/vector_rank` 表示通道内合并后的排名。原始分数与路线 RRF 分开保存：`lexical_score` 取该候选最高排名的关键词路线之原始 BM25（并记录路线 ID）；`vector_score` 优先取原问题路线的原始相似度，仅在候选只被精简路线找到时取该路线原始相似度并记录来源。`fused_score` 继续表示两通道混合 RRF。

现有 `EvidencePolicy.rejection_reason()` 用 `vector_score < 0.35` 等阈值；若将约 0.016 的 RRF 填入该字段，会误拒纯向量候选。因此分数字段的类型、来源和阈值尺度必须进入测试。精简路线分数分布需独立评估，未校准前保持该实验关闭。

所有现有 dataclass 的旧字段顺序和默认值保持可读，新字段附加在末尾。`QueryAnalysis` 同步添加同名片段、变换和路线字段；`QueryInterpreter` 显式传递，禁止仅扩展 `QueryPlan` 而遗漏分析结果。

### 4.2 正文比较结果

扩展 `rag/content_identity.py`：

```python
@dataclass(frozen=True, slots=True)
class ContentComparison:
    status: str
    # exact / layout_equivalent / suspected_equivalent / fact_conflict / different
    canonical_similarity: float
    protected_facts_left: tuple[str, ...]
    protected_facts_right: tuple[str, ...]
    differing_fragments: tuple[str, ...]
    reason: str
```

提供统一入口：

```python
def compare_content(left: str, right: str) -> ContentComparison: ...
```

`protected_facts` 至少包含：

- 含字母和数字的型号、故障码、部件编号；
- 整数、小数、百分比和数值范围；
- 单位及其相邻数值；
- `±`、`≤`、`≥`、`<`、`>`、`=`；
- “不、未、无、禁止、不可、仅、除外”等否定或范围限定词。

保留 `canonical_content()` 的入口，增加 `normalizer_version` 和保护策略，不能把旧输出当成可靠等价证明。先识别并保护技术 token、表格边界与方向符号，再进行排版规范化；单位大小写、指数、相邻数字边界不得被抹除。

代码审查中的最小运行复现：旧函数对 `5 mA / 5 MA`、`2 m² / 2 m2`、`1 2 / 12`、`A→B / AB` 均返回相同规范化结果。这四组必须成为“禁止自动等价”的回归用例。保留旧 normalizer 用于解释历史结果，新键不能混用旧缓存，也不能改写既有 stable_chunk_id。

采用 `canonicalize_with_offsets(text, structure=None)` 返回规范化文本、原文区间映射和歧义标记。中文 PDF 断行“清\n洗液”可消除；连续数字、英文 token 和表格单元格的分隔不可一律删除。缺少结构信息且无法证明是排版差异时进入疑似，避免丢弃行列关系。受保护 token 对齐比较；不同集合只能说明有差异，不能独自证明事实矛盾。

### 4.3 Gold 映射结果

扩展 `rag/evaluation/evidence_matcher.py::GoldEvidenceMatch`：

```python
@dataclass(frozen=True, slots=True)
class GoldEvidenceMatch:
    evidence_id: str
    kind: str
    file: str
    page: str
    quote: str
    chunk_ids: tuple[str, ...]   # 保留旧构造参数；只作关联索引，不能据交集直接计分
    confidence: str
    reason: str = ""
    declared_chunk_ids: tuple[str, ...] = ()
    equivalent_chunk_ids: tuple[str, ...] = ()
    suspected_chunk_ids: tuple[str, ...] = ()
    conflict_chunk_ids: tuple[str, ...] = ()
    alternatives: tuple[EvidenceAlternative, ...] = ()
```

旧 `GoldEvidenceMatch(..., chunk_ids=..., confidence=...)` 构造继续可用；新数据通过工厂生成上述分类和 `alternatives`，旧数据经版本适配器进入新评分。不能把 `chunk_ids` 改成只读属性，否则当前测试和构造调用会直接报错。

新增证据覆盖结构，避免跨块引文只拿到一半也计为完整命中：

```python
@dataclass(frozen=True, slots=True)
class EvidencePart:
    chunk_id: str
    quote_start: int
    quote_end: int             # 统一规范化 quote 的字符区间
    source_start: int
    source_end: int            # 对应原始 chunk 区间


@dataclass(frozen=True, slots=True)
class EvidenceAlternative:
    alternative_id: str
    match_type: str            # declared / layout_equivalent / reviewed_alternative
    parts: tuple[EvidencePart, ...]
    confirmed: bool
```

一个 alternative 内各 part 按 quote 区间联合覆盖（AND）；不同 alternative 之间任选一个完整覆盖（OR）。不得把两个不相关文档各半句拼成一个完整命中。见 10.3 的计分定义。

## 5. 查询解析改造

### 5.1 修改文件

- `local_full_text_search/rag/query_analyzer.py`
- `local_full_text_search/rag/query_interpreter.py`
- `local_full_text_search/rag/models.py`
- `tests/rag/test_query_analyzer.py`
- `tests/rag/test_query_constraint_safety.py`

### 5.2 QueryAnalyzer 的处理顺序

```python
def analyze(self, text: str) -> QueryAnalysis:
    original = str(text or "")
    normalized, offset_map = self._normalize_with_offsets(original)
    spans = self._extract_lossless_spans(original, normalized, offset_map)
    searchable, transformations = self._strip_question_shell_with_trace(normalized)
    constraints = self._extract_explicit_constraints(normalized, spans)
    lexical_variants = self._build_lexical_variants(searchable, spans)
    semantic_variants = self._build_semantic_variants(original, searchable, spans)
    return QueryAnalysis(...)
```

`_extract_lossless_spans()`：

1. 识别字母数字组合标识符，但只增加形态特征，不直接生成 `MODEL`。
2. 识别数字和单位，保留 `140±5 μL/s` 等完整表面文本。
3. 按标点和疑问边界提取连续文本片段；“片段”不等于已正确理解的专业名词。P0 不承诺无需分词或语法模型就识别全部名词短语。
4. 允许片段重叠，例如 `样本针水平零位传感器` 与其中的 `水平零位传感器` 可以同时存在。
5. `P17`、`SV99`、`J8` 等从未登记过的标识符也必须保留。

`_strip_question_shell_with_trace()`：

- 使用有位置记录的问句模板处理，例如句首“请问”、句尾“有哪些”，以及“是什么＋待询问名词”的中间“是什么”；仅句尾删除无法处理“P02 是什么泵”。模板需保留后面的“泵”。
- “作用、原理、校准、排查、流量、故障”等属性或业务词继续保留。
- 每次变换写入 `QueryTransformation`。
- 清理后为空或信息明显减少时，回退原始规范化问题。
- 宽召回保留清理后连续文本内部的 3/4 字窗口兼容 trigram，但不得跨问句外壳、否定、数字或标点边界；完整片段与窗口分别标记。直接删除全部窗口可能降低无编号中文问句的召回。

原文 offset 约定：`original_text[start:end] == span.text`。全角字符、Unicode 展开、空白清理通过 offset_map 回溯；规范化文本的下标不能直接套在原文上。

### 5.3 条件安全规则

1. 删除 `_STAGE_PATTERNS["sample_test"]` 中的裸词“样本”“标本”。
2. “样本检测”“标本检测”可记录为显式文本属性；无法绑定对象或存在否定/引用时仅作软提示。
3. `_SCOPE_PATTERNS/_STAGE_PATTERNS/_DIRECTION_PATTERNS` 不再仅凭词命中产生硬条件；明确肯定、可定位且能绑定同一对象的条件可交给局部冲突检查。不能把所有明确条件一律降成软条件而失去冲突保护。
4. 标识符不再全部生成 `ConstraintKind.MODEL`。
5. 无法确定角色的标识符仍进入精确检索信号。
6. `missing_fields` 不能仅因为未识别到固定阶段就要求用户补充阶段。
7. 检索前硬过滤只使用 UI 明确选择的根目录、扩展名等；检索后明确、可定位的局部事实冲突仍可拒绝。

示例预期：

```text
问题：BS-2800 生化分析仪的 P02 是什么泵？

保留片段：BS-2800、BS-2800 生化分析仪、P02、泵
移除问句外壳：是什么
硬业务条件：无
精确查询分支：BS-2800 AND P02
短语查询分支：BS-2800 生化分析仪；P02 泵
宽查询分支：上述信号的 OR 组合
向量原始查询：原问题
向量精简查询：BS-2800 生化分析仪 P02 泵
```

精简查询只允许在安全模板内删除问句外壳，并保持剩余语序和作用域，禁止加入“废液泵”等答案猜测。包含否定、更正、引文、多对象比较或无法判断关系的问题，P0 跳过精简路线；“不是 CL-900i，而是 CL-8000i”不得变成两个型号的无关系并列。

## 6. 关键词召回改造

### 6.1 修改文件

- `local_full_text_search/rag/retrieval/lexical.py`
- `local_full_text_search/rag/index_store.py`
- `local_full_text_search/rag/models.py`
- `local_full_text_search/rag/trace.py`
- `tests/rag/test_index_store.py`
- 新增 `tests/rag/test_lexical_retrieval_routes.py`

### 6.2 三条是查询路线，不是三类关键词

`QueryPlan.lexical_variants` 默认生成以下查询路线：

| route_id | 构造方式 | 目的 |
|---|---|---|
| `lexical_precise` | 两个或以上原文标识符共同出现，最多取 3 个；作用域不明时跳过 | 找到型号与部件编号共同出现的内容 |
| `lexical_phrase` | 原文中的完整连续短语 | 避免任意 3/4 字片段主导查询 |
| `lexical_broad` | 所有有效信号 OR | 保证精确路线失败时仍能宽召回 |

路线数量可以为 1、2 或 3；没有足够信号时不生成精确路线。这些路线描述执行策略，不规定关键词只能属于三个业务类别。表达式相同的路线合并为一次执行，避免同一查询重复投票。

P0 每题最多 3 条关键词路线、每路线最多 20 个去重词项；先保留原文标识符和长片段，再按原文顺序补窗口。被预算裁去的词项必须记录。这个预算属于工程限制，不意味着问题只有 20 个有效词。FTS 字面量转义、AND/OR 括号优先级和 SQL 参数绑定都需有测试；未知词不得依赖领域词表才进入查询。

### 6.3 执行和通道内合并

新增内部返回结构：

```python
@dataclass(frozen=True, slots=True)
class LexicalRouteResult:
    route: RetrievalQueryVariant
    candidates: tuple[EvidenceChunk, ...]
    duration_ms: int
```

执行流程：

1. 每条路线分别查询 FTS。
2. 每条路线最终输出最多 `lexical_top_k`。优先在查询端施加范围过滤；不具备前置过滤能力时，使用稳定排序和分页补取，扫描上限初始沿用旧路线 `4 * lexical_top_k`。到达上限仍不足时标记 `candidate_pool_exhausted`，不能将过滤后少量结果当成已穷尽语料。
3. 以 RRF 在关键词通道内部合并路线，生成唯一的 `lexical_rank`。
4. 把各路线原始排名写入 `lexical_route_hits`。
5. 关键词通道与向量通道融合时，关键词仍只占一个通道，避免三条路线被当成三票。

通道内 RRF 参数必须进入 `run.json`。初始使用与现有融合一致的 `k=60`，上线前通过 A/B 验证；不能把人为权重伪装成模型分数。

关键词通道最终输出仍截到 `lexical_top_k=50`；随后与向量通道各投一票。内部 RRF 的重复命中收益属于待检验设计，不保证精确路线第一名一定排在宽路线第一名前，需用困难负例验证。

### 6.4 短标识符兼容

FTS5 trigram 对长度不足 3 的 `J8` 之类标识符不可靠。实施时采用两层兼容：

1. 当前索引：使用参数化 `LIKE` 拉取包含该表面文本的候选，再按 ASCII 字母数字/标识符连接符边界复核，中文紧邻可接受；按稳定顺序分页补取到输出预算或扫描上限。测试 `J8接口`、`J80`、`AJ8`，不能先任意 LIMIT 后把正确候选永久截掉。FTS 表达式非空但其中短词无法被 trigram 查询时，也必须执行此补充分支。
2. 若性能测试发现全表扫描不可接受，再新增按索引构建的 `rag_chunk_identifiers(identifier, stable_chunk_id)` 表并发布新的索引 schema；旧 bundle 保持回退路径。

是否增加索引表由性能证据决定。本轮不得为了短标识符直接破坏旧 bundle 的可读性。

### 6.5 FTS 字段贡献

当前 FTS 已包含正文、文件名、路径和位置。使用 FTS5 `bm25()` 的列权重前，必须先确认建表列顺序，并把权重写入运行配置。验证内容包括：

- 只提高文件名/位置命中是否造成错误资料长期占前；
- 正文共同出现是否仍优先于仅文件名命中；
- 8 题和原 18 题是否存在回退。

未完成对照前沿用当前 BM25，先通过多路线解决召回表达问题。

## 7. 向量召回实验

### 7.1 修改文件

- `local_full_text_search/rag/retrieval/vector.py`
- `local_full_text_search/rag/retrieval/hybrid.py`
- `local_full_text_search/rag/models.py`
- `local_full_text_search/config/defaults.py`
- `tests/rag/test_chroma_vector_store.py`
- 新增 `tests/rag/test_vector_query_variants.py`

### 7.2 双查询方式

默认保留：

```text
semantic_original = 原问题
```

增加可关闭实验：

```text
semantic_focused = 删除明确问句外壳后的原文片段组合
```

如果两者规范化后相同，只执行一次。两条查询分别嵌入、分别查询 Chroma，再在向量通道内部 RRF 合并，记录 `vector_route_hits`，最终仍仅输出 `vector_top_k=50`。之后混合检索仍把向量视为一个通道。保留 `vector_score` 原始相似度，RRF 写入专用字段。

原问题路线失败按现有错误策略报告；可选精简路线失败允许降级到原问题路线，记录 `degraded_route`。取消任务必须在每次嵌入、查询及合并边界检查，不能作为失败重试。两路线的原始候选池大小和最终输出数均写入实验元数据。

新增设置建议：

```python
rag_vector_focused_query_enabled: bool = False
rag_vector_variant_rrf_k: int = 60
```

先以关闭状态合入代码并运行 A/B。满足第 15.4 节质量与耗时门槛后再修改默认值。

### 7.3 索引表示实验

如果双查询仍无法找回 003、006 等证据，再建立独立实验索引，对比：

```text
现有：chunk.text
实验：filename + location_text + chunk.text
```

该实验必须使用新的 bundle 和表示版本，不能覆盖现有索引。只有真实 8 题、原 18 题及保留题均显示净收益时才迁移。

## 8. 条件检查与证据门禁

### 8.1 修改文件

- `local_full_text_search/rag/constraint_matcher.py`
- `local_full_text_search/rag/evidence_policy.py`
- `tests/rag/test_query_constraint_safety.py`
- `tests/rag/test_rag_quality_pipeline.py`

### 8.2 行为修改

`ConstraintMatcher.evaluate()`：

- `MATCH`：候选的局部正文或可靠来源信息明确支持条件。
- `CONFLICT`：同一事实局部明确表达互斥值。
- `UNKNOWN`：正文没有说明、同时出现多个对象、或无法绑定到相同表格行。

删除以下拒绝逻辑：

```python
accepted=conflicts == 0 and not missing_device
```

改为：

```python
accepted=conflicts == 0
```

`EvidencePolicy` 取消把“推断条件未确认”直接当成全题拒答的规则；但必须与 8.3 的回答支持度判定同步交付。`accepted` 只表示候选没有明确冲突，不表示可据此给目标设备下确定结论。不得单独上线 `accepted=conflicts == 0` 这一行修改。

表格或同页包含多个型号时，模型名出现在 chunk 任意位置不能单独构成冲突。无法绑定到同一行时必须返回 `UNKNOWN`。

### 8.3 防止再次“问生化、答免疫”

增加 `support_status`（supported / unknown / conflict）及 `support_spans`；对象绑定不可靠时返回 unknown，不伪造数值置信度。

- 可证明正文属于用户询问的对象、属性及条件：用于该事实回答。
- 只找到其他设备的描述：可以作为背景保留，但不构成目标设备的确定性依据。
- 仅来源标题提示适用范围：记录标题来源及正文支持范围，不能把标题词命中当成全部事实已确认。
- 全部证据适用性未知且问题要求明确型号数值/操作步骤：返回依据不足，或在用户问题确有歧义时请求澄清；报告保留所有召回排名和不确定原因。
- 有支持证据也有背景证据：上下文逐条附支持状态，生成提示禁止把背景事实迁移到目标设备。

此部分涉及 `context_builder.py`、`answer_service.py`、现有生成请求/提示构建位置及完整问答回归。不得用评测用例的 `expected_behavior` 或 gold 参与上述决策。离线召回测试无法证明回答行为安全，必须另验真实答案和裁判结果。

## 9. 候选去重改造

### 9.1 修改文件

- `local_full_text_search/rag/content_identity.py`
- `local_full_text_search/rag/candidate_deduplicator.py`
- `local_full_text_search/rag/context_builder.py`
- `tests/rag/test_rag_quality_pipeline.py`
- 新增 `tests/rag/test_content_identity.py`

### 9.2 自动合并规则

允许自动合并：

1. `stable_chunk_id` 相同且内容版本一致；
2. `text_hash` 相同且经新版保真比较确认无事实差异；
3. 新版规范化结果相同、非空，且不存在无法恢复的结构歧义。

第三种能覆盖 PDF/PPTX/DOCX 之间可证明仅有排版差异的正文；PDF/PDF、Excel/Excel 同样适用。源表格缺少行列信息时降为疑似，不能承诺全部跨格式副本自动合并。分组表示正文身份，不等于来源设备、版本及适用范围相同；成员来源适用性仍逐个保存。合并后必须保留：

- 所有 `member_chunk_ids`；
- 所有 `SourceRef`；
- 每个成员进入各阶段的原始排名；
- 代表候选 ID 和合并原因。

### 9.3 近似正文处理

当前 `duplicate_same_document_near_text` 不再直接删除。修改为：

- `compare_content()` 为 `layout_equivalent`：合并；
- `suspected_equivalent`：两条候选均保留，记录疑似组 ID；
- `fact_conflict`：两条均保留，记录冲突事实；
- `different`：正常保留。

因此 92%/95%/99% 相似度本身不会删除候选。这样可防止一条正文只多出关键数值、否定句或解决步骤时被错误去重。

`ContextBuilder` 复用已确认的等价组及代表 ID，同时返回未进入上下文的候选及原因。确需防御性合并时使用带版本及结构保护的比较结果，不能仅把旧 `text_hash` 换成另一个未经验证的哈希键。

## 10. 统一 Gold 证据映射

### 10.1 修改文件

- `local_full_text_search/rag/evaluation/evidence_matcher.py`
- `local_full_text_search/rag/evaluation/reporting/matching.py`
- `local_full_text_search/rag/evaluation/metrics.py`
- `local_full_text_search/rag/evaluation/runner.py`
- `local_full_text_search/rag/evaluation/reporting/service.py`
- `tests/rag/test_report_matching.py`
- `tests/rag/test_rag_evaluation.py`

### 10.2 唯一判定顺序

对每条 Gold 或 hard negative：

1. 先在指定文件及页码容差内寻找引文，得到 `declared_chunk_ids`。
2. 再在其他来源寻找排版规范化后完整覆盖该引文的窗口。
3. 跨来源自动等价要求：
   - Gold quote 规范化后被候选完整覆盖；
   - 对齐的证据窗口内受保护事实一致，且正文/局部标题给出的适用对象一致或已有人工审核的替代证据关系；
   - quote 规范化长度达到可配置最小值，建议默认 24 个字符；
   - 24 字只是候选降噪参数，不证明适用性；即使 100 字完全一样，也可能来自另一型号，仍须检查局部上下文；
   - 如果 quote 太短、过于通用或适用性未知，只标为疑似。指定来源的短 quote 可在位置和上下文明确时命中，不受该跨来源长度限制。
4. 对齐窗口的相似度达到初始实验值 0.95、但仍有未解释差异时进入 `suspected_chunk_ids`。相似度比较 quote 与等长或近长窗口，不能拿短 quote 与整篇正文做 ratio。排序以 canonical 字符编辑相似度为准，明确记录算法版本。
5. 同一属性、同一局部位置存在替换型受保护事实差异时进入 `conflict_chunk_ids`。chunk 中其他句子多一个数值或型号不构成全块冲突；不能因一个含多型号表格的 token 集合不同而误判。

跨 chunk 的 Gold quote 按同一文档键、相邻 ordinal 和连续位置组成有限窗口，默认最多 3 块。重叠分块先按来源区间消除重复部分；无法确定重叠关系时不自动拼接。命中后保存 `EvidenceAlternative.parts`，每部分绑定 quote 区间和 chunk 原文区间，不能只保存一个扁平 ID 列表。

建立按 bundle ID、正文 hash、normalizer/matcher 版本缓存的规范化正文索引，优先通过精确子串/保护 token 倒排缩小候选，再做窗口比较。全语料预映射允许读取指定 bundle；离线只拿到运行结果时只匹配已记录正文，标记 `mapping_scope=recorded_candidates`，不能冒充全索引映射。所有循环支持取消并记录准备耗时。

### 10.3 评分口径

- 指定来源命中：计入 Recall/MRR，标记“指定金标”。
- 跨来源排版等价命中：计入 Recall/MRR，标记“等价金标”。
- 疑似金标：不计入正式指标，单独统计。
- 冲突证据：不计入，展示冲突事实。
- partial LCS 与原 `high >= 0.60`：均用于高亮和人工排查，v2 正式指标不能继续把 60% 覆盖称为完整命中。保留 legacy 指标用于解释历史变化；`--count-partial-matches` 只能改变单独命名的探索指标，不改变正式指标。

新增 `evaluation/stage_scoring.py`，由批量运行和离线报告共用：

```python
def score_stage(case, stage_name, candidates, mappings, *, policy):
    # 只用本阶段实际入选的正文/区间判定覆盖；不直接用 ID 交集计分
    ...
```

`RagEvalRunner.prepare()` 负责生成定位映射；`score_stage()` 负责按实际阶段覆盖计分；`reporting/matching.py` 调用同一匹配核心生成高亮，不能重新定义另一套命中标准。`metrics.py::score_case()` 与 `reporting/service.py` 均消费同一阶段评分对象，PySide6 页面也读取该对象。

完整性规则：

- 同一 alternative 的 quote 区间覆盖并集达到 100% 才计“完整证据命中”；只出现跨块证据的一部分显示“部分覆盖”，不增加完整命中数。
- 不同来源的完整 alternative 为 OR；同一 Gold 被 3 份副本命中仍只计 1 条。
- `context` 必须用 `included_text` 实际覆盖重新判定，不能因 ID 出现而把已截掉的金标句算作进入模型。
- 跨块证据的完成排名等于满足完整覆盖所需候选的最靠后名次；有多个 alternative 时取最早完成者。例如 A 在 #2、B 在 #8，A+B 才完整，则完成排名为 #8。
- 保留“首个片段名次”和“首条完整证据名次”两个字段，热力图默认完整证据名次；tooltip 说明跨块组成。
- 阶段未执行/失败用 null，阶段执行完成但未命中用 0；不得混算。

旧结果重算：不覆盖原 `cases.jsonl`、原报告或原索引。`make_report.py --index-bundle` 当前仅记录用途、未接入索引读取；需要实现只读加载并严格验证 bundle ID，才能进行全索引重映射。没有该参数时使用已保存正文匹配，明确缺少的来源和映射范围。重算结果写新目录，保留旧 matcher/metric 版本和差异原因。

### 10.4 指标分母与计分资格

正式分母固定为 reviewed、非 refuse 且 Gold 非空的有效题目，以及各题声明的 Gold 条数；不能只按“已定位成功的 Gold”缩小分母。定位失败仍显示缺失，并单列索引覆盖率。invalid/pending/disputed 排除并列出原因；阶段失败另报失败率与有效分母，不包装成检索未命中。

对执行成功的合格题集合 Q：

```text
Hit@k = 前 k 名内至少完整覆盖一条 Gold 的题数 / |Q|
MRR = sum(1 / 首条完整 Gold 的完成排名；未命中为 0) / |Q|
宏平均证据覆盖率 = sum(每题完整覆盖 Gold 数 / 本题声明 Gold 数) / |Q|
微平均证据覆盖率 = 所有题完整覆盖 Gold 总数 / 所有题声明 Gold 总数
```

MRR 显示 `倒数和/有效题数`，不是伪造整数命中分数；例如 `1.50/3 = 0.50`。同时报告宏/微平均，不能用一个累计分子分母冒充宏平均。refuse 只显示行为/风险/负例/真实分数，排除所有正向 Hit、Recall、MRR；clarify 的检索指标不代表已正确澄清。

显式区分 `union_recall_L50_V50`（两路各 50 的并集，最多 100 个）和 `fusion_recall_at_50`（固定 RRF 排序前 50）。后者从融合原始排序快照计算，即使生产只保留 30 个去重候选也不能混用。旧报告只有裁剪快照而没有足够记录时标为不可重算。

版本更新：

```python
REPORT_SCHEMA_VERSION = 4
REPORT_MATCHER_VERSION = "evidence-equivalence-v2"
REPORT_METRIC_VERSION = "diagnostic-v2"
```

## 11. 全阶段去向记录

### 11.1 修改文件

- `local_full_text_search/rag/trace.py`
- `local_full_text_search/rag/answer_service.py`
- `local_full_text_search/rag/context_builder.py`
- `local_full_text_search/rag/evaluation/runner.py`

新增统一事件结构：

```python
@dataclass(frozen=True, slots=True)
class CandidateTransition:
    chunk_id: str
    from_stage: str
    to_stage: str
    status: str
    reason_code: str
    reason: str
    rank_before: int | None
    rank_after: int | None
    representative_chunk_id: str | None = None
```

允许的 `reason_code`：

```text
present
merged_layout_equivalent
suspected_duplicate_kept
fusion_top_k_cutoff
reranker_top_k_cutoff
reranker_not_selected
constraint_conflict
below_score_threshold
document_limit
final_top_k_cutoff
context_budget_cutoff
context_text_truncated
stage_failed
unknown_legacy_result
```

各阶段必须写明实际输入数、输出数和配置上限。`AnswerService` 在取：

```python
rerank_input = deduped[:query.reranker_top_k]
```

时，为余下候选逐条记录 `reranker_top_k_cutoff`。`ContextBuilder` 遇到字符预算不足时，记录所有未进入上下文的证据，不再只执行 `break`。

现有 `final_rank` 保留为最终入选证据排名，`evidence_rank` 如新增则必须为其兼容别名；上下文使用 `context_rank`。只有一对一纳入且未重排、未删除、未合并时数字才应相同。若最终 #2 被排除，原 #3 变成上下文 #2 是合理压缩，显示 `final #3 → context #2（前项未纳入）`。等价合并显示代表和成员去向。禁止为了让图好看而强行保持不真实的名次。

去重保留条数不是融合输入排名：输入 #31 如果前面有重复，仍可能成为第 30 个保留项。去重迭代结束后按真实 dispositions 生成截断事件，不能用 `fusion_rank > 30` 直接判断。融合原始并集裁剪上限 `2 * fusion_top_k` 与去重输出上限 `fusion_top_k` 分别记录。

每阶段存 `execution_status`、`input_count`、`retained_count`、`rejected_count`、`recorded_count`、`configured_limit`。条件检查中拒绝项可展示但不占入选排名；截断项继续存在于审计区。聚合图只对真实丢失计数，“已并入仍保留”是去向事件，不计为 Gold 丢失。

## 12. HTML 诊断报告改造

### 12.1 修改文件

- `local_full_text_search/rag/evaluation/reporting/service.py`
- `local_full_text_search/rag/evaluation/reporting/navigation_assets.py`
- `local_full_text_search/rag/evaluation/reporting/timing.py`
- `tests/rag/test_report_generation.py`
- `tools/check_report_navigation.mjs`

### 12.2 查询解释区

每题显示：

- 原问题；
- 问句清理后的文本；
- 保留片段、字符位置和开放特征；
- 删除内容及原因；
- 每条关键词查询路线的真实表达式、候选数、耗时；
- 每条向量查询路线的文本、候选数、耗时；
- 生效的 UI 范围过滤和业务条件；
- 关键词及向量通道内部合并后的排名。

报告只能展示底层实际记录的信号、排名和分数，不能虚构“某词加 3 分”。

### 12.3 Gold 与事实覆盖

每条 Gold 显示：

```text
以下为展示格式示例，不是本轮重跑结果：
指定来源：资料A.pdf，第 19 页
实际命中：资料A.pptx，第 19 张
判定：跨格式排版等价
阶段轨迹：关键词 2/50 → 向量 35/50 → 融合 4/60 → ...
```

疑似等价显示规范化相似度、不同片段和受保护事实对比，并提供“疑似金标”徽章；默认不计分。

复用现有 `answer_checks` 的 id/text/role，不另建 `required_fact_checks`。增加与检查项 ID 对应的可选证据关系（下面为结构示例，实际须按已审核数据填写）：

```json
{
  "evidence_links": [
    {"check_id": "C1", "any_of": [["positive:1"]], "review_status": "pending"},
    {"check_id": "F1", "any_of": [["positive:2"]], "review_status": "pending"}
  ]
}
```

`gold_loader.py` 保留现有 `answer_checks` 及 core/safety/context 语义，增加独立 `evidence_links` 字段；`any_of` 内层为必须共同满足的证据，外层为替代组合。没有映射显示“未关联”，自动建议保持 pending，不改变整题已审核状态或裁判计分。旧 `required_facts` 回退仍沿用 `answer_judge.py::_checks()`。

当前 007/008 已有 `acceptable_alternative_evidence`，必须保留和展示，不能忽略。只有现有数据已明确关联到某条事实/Gold 且人工审核通过的替代才可做 OR；未关联的替代展示“待确认对应事实”。其语义等价来源与程序自动确认的排版等价来源分别标记。

“位置依据命中、作用依据缺失”首先描述证据类型；检索已找到相关证据不等于 LLM 答案已表达该事实。答案一致性继续由独立裁判按已有 answer_checks 判定，保留模型、提示版本、逐项理由和失败状态。

### 12.4 热力图和去向

热力图单元格继续使用：

```text
名次 / 本阶段实际候选数
```

状态区分：

- `1/50`：指定 Gold；
- `2/50 等价`：跨来源等价 Gold；
- `疑似 7/50`：疑似 Gold，不计分；
- `未命中/50`：在本阶段记录范围内没有出现；
- `已截断`：上一阶段出现，但因明确 TopK 或预算没有进入；
- `未执行`：阶段未运行；
- `执行失败`：阶段报错；
- `原因未知`：旧结果缺少去向记录。

点击单元格直接切换到题目、阶段和名次。左侧固定导航保留，并增加“仅看 Gold / 负例 / 疑似 / 截断”的筛选。

### 12.5 总览新增图表

1. 查询路线命中对比：精确、短语、宽召回、原向量、精简向量各自命中题数。
2. Gold 消失原因分布：未召回、融合截断、去重合并、重排入口截断、条件冲突、分数门槛、上下文预算。
3. 证据类型分布：指定 Gold、等价 Gold、疑似 Gold、负例、相似未标注。
4. 各模块耗时：问题准备、关键词各路线、向量各路线、融合、去重、重排、条件检查、上下文、生成、裁判。

图表使用内联 CSS/JS，不引用 CDN，继续输出单文件离线 HTML。

## 13. 分阶段开发顺序

### 阶段 A：建立可复现基线

1. 保存当前 8 题报告目录和 `run.json`。
2. 记录仓库当前提交、相关 dirty 文件的内容 SHA-256、设置、索引 bundle、模型 revision 和各 TopK；仅记 dirty 文件名不能复现。保存限定范围的补丁与未跟踪代码副本，排除凭据和用户原始资料，不执行清理/回滚。
3. 增加实验开关，所有新策略初始可关闭。
4. 不修改现有 Gold 审核状态。

交付：基线元数据和对照表。

### 阶段 B：统一正文比较和 Gold 映射

1. 扩展 `content_identity.py`。
2. 先仅实现评测使用的新版保真比较；运行时去重改动另开实验开关，不能混入“只修正评分”的对照。
3. 统一 `evidence_matcher.py` 和报告评分口径。
4. 重新生成当前保存结果的 HTML，先测量“只修正评分”带来的变化。

交付：matcher-only 报告。该报告不宣称检索质量提升。

### 阶段 C：安全查询表示

1. 增加 `QuerySpan/QueryTransformation/QueryVariant`。
2. 删除裸词阶段门禁和“所有标识符都是型号”的推断。
3. 保存真实查询变换与执行表达式。
4. 运行解析和约束安全测试。
5. 先补齐统一阶段事件、代码版本和有效分母记录，以便后续每组实验都可追溯。

交付：query-plan-v3 诊断快照。

### 阶段 D：关键词多路线召回

1. 实现精确共同出现、完整短语和宽召回路线。
2. 实现关键词通道内合并。
3. 记录路线排名和耗时。
4. 对 8 题、18 题及未调参保留题运行检索评测。

交付：lexical A/B 报告。

### 阶段 E：向量精简查询实验

1. 实现可关闭的精简查询。
2. 在向量通道内部合并，禁止双票影响混合融合。
3. 对比质量、候选变化和耗时。
4. 根据验收门槛决定默认开关。

交付：vector A/B 报告和开关结论。

### 阶段 E2：独立验证去重与适用性改动

1. 固定查询路线，仅启用新版等价去重，观察保留近似候选是否挤占其他 Gold 的重排预算。
2. 固定前述检索配置，启用局部冲突与回答支持度逻辑，检查 unknown 候选保留及回答行为。
3. 各阶段 TopK 初始不增加。若重复候选挤占明显，作为单独预算/多样性实验，不能通过跨事实误合并来恢复指标。
4. 补齐 `rag_diagnostic_dialog.py`、`rag_eval_page.py` 对新状态/指标的读取，避免桌面端与 HTML 展示不同定义。

### 阶段 F：全阶段去向与报告

1. 补齐候选转移原因。
2. 增加 Gold 类型、事实覆盖和查询路线展示。
3. 增加总览图表、筛选和跳转。
4. 验证离线打开及大报告性能。

交付：最终 `report.html`、`report_cases.csv`、`report_detail.json`。

## 14. 详细测试方案

### 14.1 单元测试：查询解析

在 `tests/rag/test_query_analyzer.py` 增加：

```python
def test_unknown_identifiers_are_preserved_without_becoming_hard_models():
    for question, expected in (
        ("BS-2800 的 P17 是什么泵？", ("BS-2800", "P17")),
        ("SV99 电磁阀有什么作用？", ("SV99",)),
        ("J8 接口连接到哪里？", ("J8",)),
    ):
        plan = QueryInterpreter().interpret(question)
        assert all(value in [span.text for span in plan.spans] for value in expected)
        assert not any(
            item.is_hard and item.kind is ConstraintKind.MODEL
            for item in plan.constraints
        )
```

其他断言：

- “样本针水平零位传感器有什么作用”不生成 `stage=sample_test`。
- “不是 CL-900i，而是 CL-8000i”保留否定和两个型号原文。
- `140±5 μL/s` 不被拆坏。
- 原问题始终等于 `semantic_original`。
- 全角/Unicode 规范化后每个 span 的 `original_text[start:end]` 仍等于原片段。
- 精简查询不包含原问题未出现的答案词。
- transformations 能解释每次删除。
- “校准原理是什么”保留“校准”和“原理”。
- “P02 是什么泵”保留泵而去除模板外壳；“有什么作用”保留作用，无标识符中文问题仍具有可执行宽召回词项。
- 否定、更正、多对象问题不生成丢失作用域的精简向量路线或 AND 精确路线。

### 14.2 单元测试：正文身份和去重

```python
@pytest.mark.parametrize("left,right", [
    ("清\n洗液", "清洗液"),
    ("• 传感器作用", "传感器作用"),
    ("Ａ４００４６", "A40046"),
])
def test_layout_only_difference_is_equivalent(left, right): ...


@pytest.mark.parametrize("left,right", [
    ("流量 140±5 μL/s", "流量 115±5 μL/s"),
    ("适用于 CL-8000i", "适用于 CL-900i"),
    ("允许启动", "不允许启动"),
    ("P02 泵", "P03 泵"),
])
def test_fact_difference_is_never_auto_merged(left, right): ...
```

另测：

- 不同 PDF 的排版等价正文合为一个候选并保留两个来源。
- PPTX 与 PDF 的等价正文合并。
- 99% 相似但关键数字不同，两个候选均保留。
- 同文档 95% 相似且尾部事实不同，两个候选均保留并标疑似。
- 合并后 Gold 可以通过任一 `member_chunk_id` 命中。
- `5 mA/5 MA`、`2 m²/2 m2`、`1 2/12`、`A→B/AB` 不自动合并；测试含空格的 `140 ± 5` 和中文断行仍可安全规范化。
- 表格值相同但行列关系不同不合并；旧 text_hash 相同也必须通过新版比较。
- `suspected_equivalent` 不通过传递关系变成确认组：A≈B、B≈C 不意味着 A=C。

### 14.3 单元测试：统一证据匹配

在 `tests/rag/test_report_matching.py` 增加：

- 指定文件、页码 ±1、quote 完整覆盖为 `declared`。
- 不同文件扩展名但 canonical quote 完整覆盖为 `equivalent`。
- quote 含 PDF 换行时仍完整匹配。
- 跨两个相邻 chunk 的 quote 能映射到两个 parts；只取第一块时为部分覆盖，第一块 #2、第二块 #8 均出现时完成排名为 #8。
- 同一 Gold 有两份完整来源，命中任意一份只计 1 条；不同来源各半句不可拼成完整命中。
- chunk ID 进入上下文但 Gold 句在截断部分：context 完整命中为 0，evidence 阶段可为 1。
- 短通用 quote 跨来源只标疑似。
- 95% 相似且存在非格式文字差异只标疑似。
- 数值、型号、编号、单位、否定冲突标为 conflict。
- Gold 与 hard negative 同块时仍显示 mixed。
- 批量 `score_case()` 与 HTML 的同阶段 Gold 数和首位排名完全一致。
- 长通用句来自另一型号，仅文本相同不得自动计为等价；同表其他行有不同数值不得误伤当前正确行。
- 原 `high=0.60` 和 partial 都不计入 v2 正式完整命中，legacy 指标另列。
- 分母样例：三道正常题、完整 Gold 完成排名为 1/2/未命中，则 Hit@1=1/3、Hit@5=2/3、MRR=1.5/3=0.5；再加入 refuse、pending 后三项不变。
- 两题覆盖 1/2、1/1 时宏平均=0.75、微平均=2/3；索引定位失败不得自动缩小原 Gold 分母。
- 构造 `GoldEvidenceMatch(..., chunk_ids=..., confidence=...)` 和旧 QueryPlan 位置参数仍可读取；旧字段不直接授权新指标完整命中。

### 14.4 单元测试：关键词和向量路线

- 精确路线可找回同时包含 `BS-2800` 与 `P02` 的 chunk。
- 只含其中一个词的 chunk 仍可通过宽路线进入候选。
- `J8` 走短标识符回退且边界复核不误命中 `J80`。
- 关键词三条路线合并后候选不重复。
- 多关键词路线在混合融合中仍只贡献一个关键词通道分数。
- 原向量与精简向量分别记录排名。
- 同一候选被两个向量查询命中，不会在关键词/向量融合时获得两份向量通道投票。
- 关闭实验开关时输出与旧单向量路线一致。
- 原始向量相似度 0.8、路线 RRF 约 0.016 时，EvidencePolicy 不得拿 0.016 和 0.35 比较；门槛测试必须检查字段来源。
- 多路线单通道输出最多 50；filter 后不足自动补取或明确记录到达扫描上限。
- 可选向量路线超时有降级记录，用户取消不会触发重试；相同表达式只执行一次。

### 14.5 集成测试：阶段去向

在 `tests/rag/test_rag_quality_pipeline.py` 和 `tests/rag/test_rag_end_to_end.py` 验证：

- 当前已有 30 条不同正文被保留时，下一条不同正文才记录 `fusion_top_k_cutoff`；若前 30 个输入包含重复，输入第 31 名仍可被保留。
- 去重后第 21 名在 `reranker_top_k=20` 时记录 `reranker_top_k_cutoff`。
- 明确冲突记录候选原文和 `constraint_conflict`。
- 条件未知不会从候选池删除；只有免疫来源却问生化数值时，不得因候选保留就输出确定的生化答案。
- 最终第 9 名记录 `final_top_k_cutoff`。
- 上下文预算不足记录 `context_budget_cutoff` 或 `context_text_truncated`。
- 最终 #2 被预算排除后，原 #3 可成为上下文 #2，必须记录压缩原因；无删除/合并/重排时排名不变。
- 使用真实 Chroma 临时数据库，但 Embedding/Reranker/LLM 测试替身的范围必须写在测试说明中。

### 14.6 报告测试

在 `tests/rag/test_report_generation.py` 验证 HTML/JSON/CSV：

- 包含 query-plan-v3 和 matcher 版本。
- 热力图显示 `1/50`、`未命中/50`、`已截断`、`未执行`。
- 指定、等价、疑似、冲突证据颜色和文字不同。
- 007 能显示“位置依据已命中、作用依据未命中”。
- 页面直接显示实际答案和拒答原因。
- 每个候选可展开全部来源和去重差异。
- 无 CDN、无外部字体、UTF-8。
- 旧 schema 报告可读取，缺失原因标为 `unknown_legacy_result`。
- 旧结果重算不读取任意 active bundle；显式 bundle 不一致时退出码非零并显示警告，未提供索引时显示 recorded_candidates 范围。
- 金标文本含 `</script>`、引号和 HTML 标签时正确转义；浏览器离线无外部请求；CSV 内容按文本处理，避免公式注入。
- 测试仅断言数据和行为，不只断言 HTML 中存在某段文字。

扩展 `tools/check_report_navigation.mjs`：

- 点击热力图切到正确题目、阶段、名次；
- 筛选“等价 Gold”“疑似 Gold”“截断”；
- 导航隐藏后可恢复；
- 切换阶段后保留用户展开/收起状态；
- 1600×900 和 1366×768 下固定导航不遮挡内容。

## 15. 真实数据 A/B 验证

### 15.1 对照矩阵

先用相同保存结果做 matcher-only 重算；运行时去重和门禁不能悄悄混入查询收益。所有比较统一使用 v2 正式分母，legacy 分数只作历史旁注。

| 组别 | 相对前一组唯一变化 | 固定内容 | 要验证的原因 |
|---|---|---|---|
| A 基线 | 无；旧结果保留 | 原始结果与 legacy 指标 | 历史事实 |
| B 评分修正 | 用 v2 重算同一份候选快照 | 全部检索结果不变 | 评分变化，不能宣称召回改善 |
| C1 解析 | 启用 query-plan-v3，词项仍走旧单 OR 执行 | 原向量/去重/门禁 | 解析和词项准备影响 |
| C2 关键词 | 启用关键词多路线 | C1 查询表示，其余同 C1 | 查询执行方式影响 |
| D 向量 | 启用精简向量实验 | 关键词、去重、门禁固定 | 补充向量路线收益 |
| E 去重 | 启用新版确认等价/保留疑似 | 查询路线固定 | 去重及候选挤占影响 |
| F 适用性 | 局部冲突＋回答支持度同时启用 | 检索、去重固定 | 条件检查和答案行为影响 |
| G 验收 | 冻结通过门槛的最终组合 | 预先声明配置 | 未调参数据与完整问答验收 |

每组使用显式 `experiment_config` 注入 worker/runner/service，记录生效配置，不能修改全局用户设置后忘记还原。旧基线运行无法重新构建时使用保存快照，标记不可比较的性能项，不声称相同代码环境重跑。C1 的旧门禁兼容层仍使用旧 constraints，只替换检索词项；新条件策略仅在 F 启用，否则无法隔离变量。

每组保存独立运行目录，`run.json` 必须记录：

- Git commit、相关 dirty 文件内容 hash 和可恢复的限定范围代码快照；
- dataset SHA-256；
- bundle ID；
- QueryPlan/matcher/metric 版本；
- 每个实验开关；
- 所有 TopK 和 RRF 参数；
- Embedding/Reranker revision；
- 各阶段耗时。

### 15.2 数据集

1. 当前 8 道跨领域审核题：`rag_gold_cross_domain_v1.jsonl`。
2. 已有 18 道审核题：`rag_gold_v1.jsonl`。
3. 至少增加一组不参与开发调试的保留题，包含未见过的型号、子模块编号、否定条件、表格多型号和跨格式重复来源。

保留题在代码冻结前不用于调整词表、阈值或权重。自动生成的新题默认 pending，只有用户逐题核实后才进入正式计分；已有人工 reviewed 的 8/18 题照常验证。保留题尚未审核时只能报告开发集结果，不能宣称泛化验收通过。

8/18 题必须核对各自 bundle 与快照；不同 bundle 的运行分开记录，不自动改 corpus_snapshot 来通过检查。向量表示实验如产生新 bundle，记录受控的原始语料对应关系和迁移审核，不能直接绕过版本校验。

### 15.3 每题核对

对 8 题输出：

- 指定 Gold、等价 Gold、疑似 Gold 数量；
- 关键词各路线排名；
- 向量各路线排名；
- 融合、去重、重排、条件、最终证据、上下文排名；
- 首次消失阶段和原因；
- 必需事实覆盖；
- 各模块耗时。

特别验收：

| 题号 | 必查结果 |
|---|---|
| 003 | `BS-2800` 与 `P02` 均被保留，`P02` 不成为硬设备型号 |
| 004 | PDF/PPTX 内容满足规则时显示“等价 Gold”，实际来源和各阶段名次可追溯 |
| 007 | 不生成 `sample_test` 硬条件；明确显示缺失的是“作用”依据 |
| 008 | Gold 若在重排入口前被截断，显示准确上限和截断名次 |

### 15.4 指标与门槛

必须同时报告：

- 指定来源 Recall/MRR；
- 指定 + 自动等价 Recall/MRR；
- 疑似证据数量；
- hard negative 数量和最高排名；
- `union_recall_L50_V50` 与 `fusion_recall_at_50`，分别说明候选预算；
- 进入 Reranker 的必需证据覆盖；
- 最终上下文必需事实覆盖；
- 单题各模块 P50/P95 耗时。

启用新策略的最低门槛：

1. 与 B 相同评分口径相比，8 题的融合前 50 完整 Gold 覆盖不下降，且至少一个当前失败题得到真实召回改善；仅重算评分不算召回改善。同时报告两路并集，避免只取有利指标。
2. 原 18 题以同一 matcher、同一 bundle、同一阶段预算比较，原本正确的上下文证据和 core/safety 答案事实不得出现未解释回退；词法/向量单路线可有名次变化，必须逐题报告。发生核心证据或答案回退时修复或保持对应实验关闭。
3. 所有数值、型号、编号、单位和否定冲突测试为 0 个错误自动合并。
4. 条件 `UNKNOWN` 不得自动造成候选删除，也不得被当成已支持事实；“问生化、答免疫”的完整问答回归必须通过。
5. 双向量查询只有获得独立质量收益才考虑默认启用。性能按同机、同 bundle、同缓存条件交替 A/B，预热后至少 5 轮，另列冷启动。8 题单轮的 P95 不足以作稳定结论：逐题报告中位数、范围、P95 估计及样本数，计入 query embedding 和准备时间。10% 是待验证工程预算而非统计显著门槛；结果波动覆盖预算边界时保留关闭。
6. 报告中的阶段指标与 `report_detail.json` 逐题重新计算结果完全一致。

多路线会增加原始扫描量，即使最终仍为每通道 50 条，也不能声称算力预算相同。A/B 同时报告最终 TopK、后端拉取量、embedding 次数、扫描截断率；需要时增加相同总候选池预算的对照，区分查询质量与单纯扩大搜索量。

“8/8 全命中”保留为优化目标，不作为通过错误扩展等价证据或修改 Gold 达到的硬指标。

### 15.5 完整问答验收

检索验收后固定上下文、模型/提示版本及配置，运行 8+18 reviewed 题的 full 模式，沿用独立 LLM 裁判和 3 次瞬时错误自动重试。记录答案、拒答/澄清原因、core/safety/context 逐项结果、API 次数和失败状态。

重点人工抽查：生化/免疫同名模块、型号不同但正文相似、未知适用性、明确冲突、拒答和澄清。禁止只凭裁判总分认定问题已修复；裁判失败单列，不当作“不一致”。retrieval-only 显示“未生成答案/未调用裁判”，不能把它的预备状态展示成真实回答。

## 16. 建议执行命令

语法与目标测试：

```powershell
.venv\Scripts\python.exe -m py_compile `
  local_full_text_search\rag\models.py `
  local_full_text_search\rag\query_analyzer.py `
  local_full_text_search\rag\query_interpreter.py `
  local_full_text_search\rag\content_identity.py `
  local_full_text_search\rag\candidate_deduplicator.py `
  local_full_text_search\rag\evaluation\evidence_matcher.py `
  local_full_text_search\rag\evaluation\reporting\matching.py

.venv\Scripts\python.exe -m pytest -q `
  tests\rag\test_query_analyzer.py `
  tests\rag\test_query_constraint_safety.py `
  tests\rag\test_index_store.py `
  tests\rag\test_report_matching.py `
  tests\rag\test_report_generation.py `
  tests\rag\test_rag_evaluation.py `
  tests\rag\test_rag_quality_pipeline.py `
  tests\rag\test_rag_end_to_end.py
```

新增测试文件实现后，再执行（以下三文件为本方案待新增项）：

```powershell
.venv\Scripts\python.exe -m pytest -q tests\rag\test_content_identity.py tests\rag\test_lexical_retrieval_routes.py tests\rag\test_vector_query_variants.py
```

还需按受影响范围回归现有 `test_answer_judge.py`、`test_generation_gateway.py`、`test_rag_diagnostics.py`、`test_rag_eval_page.py` 和 `test_rag_query_worker_diagnostics.py`。保存退出码、passed/failed/skipped 数及跳过原因；不能把 chromadb 或真实模型缺失导致的 skip 计为端到端通过。

真实本地模型测试需明确启用，不能把测试替身结果描述成真实模型结果：

```powershell
$env:LFTS_TEST_REAL_RAG_MODELS = "1"
.venv\Scripts\python.exe -m pytest -q tests\rag\test_local_models_integration.py
```

从保存结果重建离线报告：

以下使用已保存候选模式；正式实现 v2 后会标记 `mapping_scope=recorded_candidates`，不能据此声明索引中不存在其他证据。

```powershell
.venv\Scripts\python.exe make_report.py `
  --cases tests\rag\fixtures\rag_gold_cross_domain_v1_review.json `
  --results C:\Users\50263194\AppData\Local\LocalFullTextSearch\rag_eval_runs\20261026_113214_421f57 `
  --out reports\cross_domain_matcher_v2 `
  --topk 50
```

真实 8 题检索评测继续通过应用的“RAG 批量评测”页面执行，模式选择 `retrieval-only`。运行前必须确认页面显示的 bundle ID 与测试集 `corpus_snapshot` 一致。

为避免各组实验手工改全局设置，实施时新增 `tools/run_rag_experiment.py`（当前尚不存在）：从指定配置 JSON 构造 settings 的内存副本，复用 `RagEvalRunner` 和现有本地模型/服务工厂。参数契约：`--cases`、`--experiment-config`、`--mode retrieval-only|full`、`--out`；配置中显式记录 generation_base_path、期望 bundle、查询/去重/门禁版本及开关。输出各组 manifest 和原始逐题记录，错误返回非零退出码。配置不得包含 API Key；full 模式沿用凭据库与脱敏流程。

命令示意（入口和 experiment-config 在开发时创建后方可执行）：

```powershell
.venv\Scripts\python.exe tools\run_rag_experiment.py --cases tests\rag\fixtures\rag_gold_cross_domain_v1.jsonl --experiment-config reports\rag_query_v3\configs\C2.json --mode retrieval-only --out reports\rag_query_v3\C2
```

## 17. 交付物与完成定义

开发完成时交付：

1. 上述代码修改及中文注释。
2. 单元、集成和报告导航测试结果。
3. A/B、C1/C2、D/E/F/G 对照结果或无法运行某组的明确原因；未运行项明确标为未验收。
4. 新旧 `report.html`、`report_cases.csv`、`report_detail.json`。
5. 一份逐题差异表，区分：
   - 原报告漏计分；
   - 查询/召回真正改善；
   - 排名改善但仍被后续截断；
   - 仍未解决；
   - 疑似证据等待人工审核。
6. 性能对比，尤其说明 Cross-Encoder 输入量是否增加。
7. 明确验证范围：规则单测、测试替身链路、真实本地模型链路、真实 LLM 链路分别列出，不互相替代。

完成定义：用户能从报告中直接回答以下问题，并能跳转到对应候选正文：

- 本题实际使用了哪些关键词和向量查询？
- 指定 Gold、等价 Gold、疑似 Gold 和负例分别排第几？
- Gold 在哪个阶段第一次消失？
- 消失是未召回、合并、截断、冲突、分数门槛还是上下文预算造成？
- 找到的是对象/位置依据，还是足以回答“作用、原理、数值”的事实依据？
- 指标上涨来自评分口径修正，还是检索本身真正改善？

只有这些问题都有真实记录可追溯，且对照测试满足门槛，本轮改造才算完成。

## 18. 第二轮方案审查记录（2026-10-26）

本节记录方案缺陷及修订，不表示业务代码已经修复。已只读核对实际类、调用点和保存报告，并运行正文规范化的四组最小反例；未进行全量测试或真实问答。

| 优先级 | 原方案问题 | v1.1 修订 | 开发验收证据 |
|---|---|---|---|
| P0 | 规范化相同直接合并，忽略单位大小写、指数和结构 | 版本化保真规范化、结构歧义降级 | 4 组实测反例和表格反例不能合并 |
| P0 | RRF 与 vector_score 的含义混淆 | 原始分数、路线 RRF、混合 RRF 分列 | 0.8 相似度不会因 0.016 RRF 被门槛误拒 |
| P0 | GoldEvidenceMatch 改成属性却宣称兼容 | 保留旧构造字段，新增 alternative/part | 现有关键字和位置参数构造可读 |
| P0 | 任意一个跨块 ID 出现就可命中 | 同组区间完整覆盖，跨来源 OR | 半块不计完整；#2+#8 完成排名 #8 |
| P0 | 上下文只按 ID 判定会把截断句算入 | 按 included_text 重新匹配 | evidence 命中但 context 截掉时不计分 |
| P0 | unknown 放行可能重现跨专业回答 | 候选保留与回答支持度分别判定 | 生化/免疫真实问答回归 |
| P1 | 长度≥24、内容一样即跨来源金标 | 检查适用范围/局部上下文，未知标疑似 | 长通用句不同型号不自动计分 |
| P1 | 完整短语提取、IDF、问句删除缺乏落地规则 | 开放片段、offset 映射、保留宽召回窗口和模板边界 | 无编号、短编号、否定、多对象测试 |
| P1 | 事实检查新增一套结构，忽略已存在替代证据 | 复用 answer_checks，单独 evidence_links | 原裁判 core/context 语义保持，替代关系可追溯 |
| P1 | 融合 #31 必截断、最终与上下文排名必相同 | 按真实保留数和转移事件判定 | 重复项、过滤项和预算压缩反例 |
| P1 | 比较实验同时改多个因素，8 题单轮 P95 作硬门槛 | 独立消融、相同评分口径、多轮时延 | C1/C2/D/E/F 每步变化可归因 |
| P1 | Recall/MRR 分母及失败/拒答未固定 | 第 10.4 节定义正式口径与资格 | 人工小数据独立核算与端间一致 |

仍需通过开发实验确定的事项：安全分词片段是否足够、关键词路线 RRF 是否产生净收益、保留近似块是否挤占候选、双向量是否值得额外成本。方案对这些保留关闭开关和失败判据，不提前写成效果保证。
