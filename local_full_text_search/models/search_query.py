from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class SearchQuery:
    """一次搜索请求的完整快照。

    教学提示：它不负责显示界面，也不负责查询数据库，只负责把用户在界面上
    选择的搜索条件一次性封装起来。这样后台线程开始工作后，即使用户继续修改
    输入框，本次搜索使用的条件也不会跟着变化。
    """

    # 用户真正输入的关键词，例如“武汉 AI”。这是唯一没有默认值的必填字段。
    text: str
    # 关键词组合方式，例如 exact、all、any、phrase 或 regex。
    mode: str = "exact"
    # 只搜索这些根目录；空列表表示不额外限制根目录。
    root_ids: list[int] = field(default_factory=list)
    # 只搜索这些扩展名；空列表表示不额外限制文件类型。
    extensions: list[str] = field(default_factory=list)
    # 下面三个开关决定关键词可以从文件名、路径和正文中的哪些来源命中。
    search_filename: bool = True
    search_path: bool = True
    search_content: bool = True
    # 是否包含 OCR 识别文字，以及是否允许低置信度 OCR 结果参与搜索。
    include_ocr: bool = True
    include_ocr_fuzzy: bool = False
    ocr_min_confidence: float = 0.60
    # 文本归一化规则：是否区分大小写、忽略空格、忽略连字符。
    case_sensitive: bool = False
    ignore_spaces: bool = False
    ignore_hyphens: bool = False
    # 可选的修改时间范围。
    date_from: datetime | None = None
    date_to: datetime | None = None
    # 可选的文件大小范围，单位是字节。
    min_size: int | None = None
    max_size: int | None = None
    # 分页与结果上限。page 是当前页，cursor 为将来/特殊查询保留游标位置。
    page_size: int = 100
    max_results: int = 1000
    page: int = 1
    cursor: str | None = None
    # 最终结果排序方式，例如 relevance（相关度）。
    sort_by: str = "relevance"
