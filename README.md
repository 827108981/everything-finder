# 本地多格式全文搜索工具

Windows 本地离线全文强匹配搜索工具。程序会先为用户指定目录建立 SQLite FTS5 索引，搜索时查询索引并做二次字符串确认，不会把文件或搜索词上传到网络。

## 已实现

- PySide6 中文桌面界面：目录管理、索引控制、搜索、卡片式分页结果、右侧预览、失败文件查看。
- 增量索引：根据路径、文件大小、修改时间跳过未变化文件；删除文件会从结果中移除。
- 首次全量扫描提速：普通文档多线程解析，OCR、ZIP、老版 Office 分流到慢任务队列，解析结果批量写入 SQLite。
- SQLite + FTS5 trigram 索引：保存文件元数据、内容块、失败状态和短词辅助表。
- 解析器：TXT/LOG/CSV/MD/JSON/XML/INI、PDF 原生文本、DOCX 段落/表格/页眉页脚、XLSX/XLSM 行与单元格、PPTX 幻灯片/备注、ZIP 内部文件、老版 DOC/XLS/PPT 转换解析。
- OCR：图片 OCR 和扫描 PDF OCR 已接入 PaddleOCR，团队版默认启用，并随包携带本地模型。
- OCR 提速策略：极小图片默认只索引元数据，超大图片会缩放后 OCR，降低无效图片和大图对全量扫描的拖累。
- 诊断状态：真实失败、部分成功、OCR 不可用、转换器缺失、仅元数据、暂不支持会分开记录，失败清单支持筛选和导出汇总。
- MP4：默认只索引文件名/路径，不作为解析失败。
- 搜索模式：精确包含、全部关键词、任一关键词、完整短语、仅文件名。
- 文件操作：双击打开文件，可打开所在文件夹。
- 交付文件：运行脚本、打包脚本、PyInstaller spec、测试、使用说明、配置说明、数据库说明。

## 快速运行

```bat
run_dev.bat
```

脚本会创建 `.venv`、安装 `requirements.txt` 并启动 `app.py`。

如果只想手动运行：

```bat
python -m pip install -r requirements.txt
python app.py
```

## OCR 团队版

OCR 是团队交付版本的默认能力。开发环境安装 OCR 依赖：

```bat
python -m pip install -r requirements-ocr.txt
```

程序会优先使用项目内 `ocr_models` 目录中的离线模型；如果 OCR 依赖或模型不可用，图片和扫描 PDF 会保留文件名/路径索引，并在诊断状态中显示原因。

## 首次扫描提速

全量扫描采用流水线：

```text
文件扫描 -> 普通解析线程池 / OCR 队列 / 慢任务队列 -> 批量写入 SQLite
```

可在设置页调整：

- 普通解析线程数：建议 4～8。
- OCR 工作线程数：建议 1～2，过高会明显占 CPU/内存。
- ZIP/老版 Office 慢任务线程数：建议 1～2。
- 批量写库文件数：默认 32。
- 小图片 OCR 跳过像素：用于跳过图标、缩略图等无搜索价值图片。
- 图片 OCR 最大边长：超大图片会先缩放再识别。

## 测试

```bat
python -m unittest discover -s tests
```

当前测试覆盖文本标准化、文本解析、数据库、增量索引、并行批量索引、搜索模式、扩展名过滤、DOCX/XLSX 解析、ZIP 解析、老版 Office 转换缺失处理、图片 OCR 跳过规则和 MP4 仅元数据处理。

## 打包

```bat
build.bat
```

输出目录：

```text
dist\本地多格式全文搜索工具\
```

如需跳过 OCR 依赖构建普通兜底包：

```bat
set SKIP_OCR=1
build.bat
```

## 默认数据位置

```text
%LOCALAPPDATA%\LocalFullTextSearch\
```

包含数据库、日志、配置、OCR 缓存和临时文件。程序不会默认把数据库写到安装目录。

## 当前限制

详见 [docs/已知限制.md](docs/已知限制.md)。重点限制：老版 DOC/XLS/PPT 需要本机安装 Microsoft Office 或 LibreOffice 才能转换；PDF 页内坐标高亮、Office 内嵌图片 OCR、正则高级限时搜索仍为后续增强。
