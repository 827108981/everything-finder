# RAG V1 使用与配置说明

## 1. 需要准备什么

- 本地搜索索引已发布，包含 PDF、DOCX 或 TXT 内容。
- Embedding 模型由程序首次构建 RAG sidecar 时下载。默认 `BAAI/bge-small-zh-v1.5`，保存到 `%LOCALAPPDATA%\LocalFullTextSearch\cache\rag-fastembed`；后续从本机缓存加载。
- 一个 OpenAI 兼容 Chat Completions API 的 Base URL、model ID 和 API key。设置页默认 Base URL 为 `https://api.openai.com/v1`，也支持其他兼容 endpoint。

OpenAI ChatGPT 订阅与 API Platform 分开计费。API key 需在 API Platform 创建并配置该平台的计费。[OpenAI 计费说明](https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform) · [API key 安全建议](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)

## 2. 在应用中配置

1. 打开“设置” → “本地智能问答”，勾选“启用智能问答”。
2. 首版使用默认 Embedding model ID；只有切换到 FastEmbed 支持列表中的模型时才修改它。改变模型 ID 后需要重建 RAG sidecar。
3. 填写 API Base URL 和 model ID。
4. 在 API Key 密码框中输入 key，点击“保存密钥”。应用通过 Windows Credential Manager 保存密钥；key 不写进 `settings.json`。
5. 点击“测试连接”。测试会使用保存的 key 请求 API 的 `/models`；若兼容服务不提供模型列表，界面会提示这一点，首次实际提问会继续验证 Chat Completions endpoint。
6. 返回“智能问答”页，点击“构建/更新智能索引”。首次构建会联网下载 Embedding ONNX 模型；成功后 sidecar 与当前搜索 generation 对齐。

智能问答请求会把问题和检索出的资料片段发送到设置页配置的 API URL。页面会在提问前显示此说明。不要把 API key 发到聊天或写入仓库。

## 3. 提问和来源

在“智能问答”页输入问题，可选搜索目录、文件类型、OCR 和最低 OCR 置信度。回答中的 `[S1]` 等标签对应下方来源卡片。点击“打开位置”会使用现有文件定位功能打开 PDF 页码或 TXT 行号；DOCX 显示索引当前保留的段落范围。

没有足够证据时，程序不请求 LLM 生成结论。若 API 返回无效引用，程序尝试一次格式修复；仍不合规时显示固定的校验失败提示和证据卡片。

## 4. 诊断与重建

手动构建 sidecar：

```powershell
$ragIndexBase = 'E:\index-store\search_index.db'
.\.venv\Scripts\python.exe tools\build_rag_index.py --index-base $ragIndexBase --model BAAI/bge-small-zh-v1.5
```

模型缓存已存在时离线构建：

```powershell
.\.venv\Scripts\python.exe tools\build_rag_index.py --index-base $ragIndexBase --model BAAI/bge-small-zh-v1.5 --offline
```

运行检索评测：

```powershell
.\.venv\Scripts\python.exe tools\run_rag_eval.py --cases evals\retrieval_cases.jsonl --modes lexical,vector,hybrid --output E:\rag-reports
```

`retrieval_cases.jsonl` 每行一条人工标注问题，结构例如：

```json
{"id":"recovery-01","question":"索引中断后怎样恢复？","relevant_sources":[{"path":"E:/demo/recovery.txt","line_start":1}],"extensions":[".txt"]}
```

如果搜索 generation 已改变，问答页会显示 sidecar 过期；重建后才能继续问答。更换 API model 或 URL 不要求重建 Embedding sidecar；更换 Embedding model 或 chunk policy 需要重建。

关闭“启用智能问答”即可停用 RAG，现有关键词搜索保持可用。删除 API key 可在设置页点击“删除已保存密钥”；删除模型缓存前关闭应用并删除上述 `rag-fastembed` 子目录。
