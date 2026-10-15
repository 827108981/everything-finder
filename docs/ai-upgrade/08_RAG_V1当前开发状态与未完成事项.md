# RAG V1 当前开发状态与未完成事项

更新时间：2026-10-15

## 当前结论

RAG V1 的主要代码已经落地，Embedding 使用本地 FastEmbed 模型，回答使用用户配置的 OpenAI 兼容 Chat Completions API。核心自动化测试通过。尚未完成真实 API 联调、完整 UI 人工验收和 Windows 发布包构建，因此当前状态是“核心实现已验证，交付验收未完成”。

## 已完成

- 增加 RAG sidecar、generation 对齐、候选校验和原子发布流程；搜索索引与 RAG 向量数据分别存储。
- 增加 chunk/evidence 数据结构、来源位置、稳定 ID、token 上限处理和 OCR 来源过滤。
- 集成 FastEmbed `BAAI/bge-small-zh-v1.5`。模型已下载到本机缓存；实际模型 revision 为 `46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59`，维度为 512。缓存模型已完成离线向量生成和检索 smoke test。
- 增加关键词/FTS、向量和 RRF 混合检索，以及来源过滤和结果多样性处理。
- 增加基于证据的上下文、无证据拒答、引用校验及一次格式修复。
- 增加 OpenAI 兼容 API 设置：Base URL、model ID、API key；key 存入 Windows Credential Manager，不写入应用设置文件。请求使用 Chat Completions streaming 和 `max_completion_tokens`。
- 增加智能问答页、设置页、后台构建/查询/API 探测 Worker、取消和来源打开流程。
- 增加 sidecar 构建脚本、检索评测脚本、配置说明和代码级方案。

## 已完成的自动验证

| 验证 | 结果 |
| --- | ---: |
| `tests/rag` | 24 项通过 |
| 现有搜索与 UI 回归：compact snapshot、SearchEngine、index UI controls、startup background snapshot | 77 项通过 |
| RAG、Worker、UI、工具脚本 `compileall` | 通过 |
| `git diff --check` | 通过；仅显示 Git 将部分 LF 文件转换为 CRLF 的提示 |
| API 请求单元测试 | 使用 HTTPX MockTransport 验证 URL、Bearer key、model、流式响应与新版 token 参数 |
| API key 存储测试 | 使用假 keyring 验证 key 不进入 AppSettings 序列化 |

真实 API key 尚未配置，以上 API 测试不代表已连接 OpenAI 或其他在线服务。

## 尚未完成及当前限制

1. **真实 API 联调**：用户尚未提供 API Base URL、model ID 和 key。需要在应用“设置 → 本地智能问答”中录入并测试连接，再实际提问验证兼容性、模型权限和返回格式。不要把 key 发到聊天或提交到 Git。
2. **完整 UI 人工验收**：尚未逐步操作设置、索引构建、筛选提问、取消任务以及点击 PDF 页码/TXT 行号/DOCX 段落来源。
3. **Windows 发布包**：PyInstaller 全量打包在分析 Paddle、ONNX Runtime 和应用依赖时被用户要求停止。该过程以 Ctrl+C 中断，没有得到成功或失败的最终构建结果，也没有可供验收的 exe。日志出现 `onnxruntime.quantization` 子模块因缺少可选 `onnx` 包而未收集的警告；是否影响实际发行版仍需构建后检查。PaddleX/Paddle 的收集警告也需要在完整包运行时判断。
4. **真实语料检索评测**：评测脚本已实现，但还没有准备人工标注的 `retrieval_cases.jsonl` 数据集，也未产生 Recall@K/MRR 报告。
5. **首装下载流程**：开发机已完成模型下载与缓存后离线加载验证；尚未在清空缓存的干净环境中验证首次下载进度、失败重试和代理/网络限制下的提示。
6. **目标 API 的参数兼容性**：当前使用 Chat Completions 的 JSON mode、streaming 和 `max_completion_tokens`。具体第三方兼容服务可能不支持 `/models`、JSON mode 或这些参数，需要用用户提供的 URL/model 联调确认。

## 下一步验收顺序

1. 在设置页配置 API Base URL、model 和 API key；点击“测试连接”，然后用一条无敏感内容的问题验证完整请求。
2. 准备一个小型 PDF、DOCX、TXT 测试目录，构建 RAG sidecar，检查 generation、来源位置、OCR 过滤和问答引用。
3. 完成人工 UI 验收并记录问题；准备至少 20 条带正确来源的检索评测用例，运行 Recall@K/MRR。
4. 修复验收中发现的问题后重新运行自动化测试，并在临时输出路径构建完整 Windows 包；在没有模型缓存的干净环境安装/启动，验证 Embedding 下载、凭据读取和 API 请求。
5. 只有完整包通过启动与核心流程检查后，才把该包作为可交付版本。

## 本次停止开发时的工作区状态

打包使用独立临时目录，没有覆盖仓库原有 `dist`。用户要求停止后，PyInstaller 进程已中断；临时目录只含未完成的分析文件。仓库中的 RAG 代码和文档尚待提交与推送。本文件记录的是截至更新时间的验证结果，不表示 Windows 安装包或在线 API 已验收。
