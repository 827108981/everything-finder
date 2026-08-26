from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 48
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X
TOP_Y = PAGE_HEIGHT - 50
BOTTOM_Y = 46

BLUE = colors.HexColor("#2563EB")
BLUE_DARK = colors.HexColor("#173B72")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#617086")
LINE = colors.HexColor("#DCE3EC")
PANEL = colors.HexColor("#F5F7FA")
PALE_BLUE = colors.HexColor("#EFF6FF")
PALE_GREEN = colors.HexColor("#ECFDF5")
PALE_AMBER = colors.HexColor("#FFFBEB")
PALE_RED = colors.HexColor("#FEF2F2")
WHITE = colors.white

VERSION = "1.5.0"
GUIDE_DATE = "2026 年 8 月 14 日"
PAGE_TOTAL = 8
SCREENSHOT_CAPTION = "当前版本真实界面，内容为演示数据"


def register_fonts() -> None:
    candidates = (
        (Path(r"C:\Windows\Fonts\Deng.ttf"), Path(r"C:\Windows\Fonts\Dengb.ttf")),
        (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\msyhbd.ttc")),
    )
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("GuideSans", str(regular)))
            pdfmetrics.registerFont(TTFont("GuideSans-Bold", str(bold)))
            return
    raise FileNotFoundError("A Chinese TrueType font is required to build the guide")


def wrap_text(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and pdfmetrics.stringWidth(candidate, font, size) > width:
                lines.append(current.rstrip())
                current = character.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
    return lines


class Guide:
    def __init__(self, output: Path, assets: Path) -> None:
        self.output = output
        self.assets = assets
        self.canvas = Canvas(str(output), pagesize=A4, pageCompression=1)
        self.canvas.setTitle("本地多格式全文搜索工具 - 图文使用说明")
        self.canvas.setAuthor("本地多格式全文搜索工具")
        self.canvas.setSubject("版本 1.5.0 的安装、索引、搜索、暂停、故障处理和数据库迁移说明")
        self.canvas.setKeywords("本地全文搜索, OCR, PDF, Office, 索引, 使用说明")
        self.page_number = 0

    def begin_page(self, title: str, subtitle: str) -> float:
        if self.page_number:
            self.canvas.showPage()
        self.page_number += 1
        self.canvas.setFillColor(BLUE_DARK)
        self.canvas.setFont("GuideSans-Bold", 19)
        self.canvas.drawString(MARGIN_X, TOP_Y, title)
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("GuideSans", 9.3)
        self.canvas.drawString(MARGIN_X, TOP_Y - 21, subtitle)
        self.canvas.setStrokeColor(LINE)
        self.canvas.line(MARGIN_X, TOP_Y - 34, PAGE_WIDTH - MARGIN_X, TOP_Y - 34)
        return TOP_Y - 55

    def footer(self) -> None:
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("GuideSans", 8)
        self.canvas.drawString(
            MARGIN_X,
            25,
            f"本地多格式全文搜索工具 · 图文使用说明 {VERSION}",
        )
        self.canvas.drawRightString(
            PAGE_WIDTH - MARGIN_X,
            25,
            f"{self.page_number} / {PAGE_TOTAL}",
        )

    def text(
        self,
        value: str,
        y: float,
        *,
        x: float = MARGIN_X,
        width: float = CONTENT_WIDTH,
        size: float = 9.5,
        leading: float = 14,
        color: colors.Color = INK,
        bold: bool = False,
    ) -> float:
        font = "GuideSans-Bold" if bold else "GuideSans"
        self.canvas.setFillColor(color)
        self.canvas.setFont(font, size)
        for line in wrap_text(value, font, size, width):
            self.canvas.drawString(x, y, line)
            y -= leading
        return y

    def heading(self, title: str, y: float) -> float:
        self.canvas.setFillColor(INK)
        self.canvas.setFont("GuideSans-Bold", 12.5)
        self.canvas.drawString(MARGIN_X, y, title)
        return y - 21

    def bullets(
        self,
        items: list[str],
        y: float,
        *,
        size: float = 9.4,
        leading: float = 14,
        gap: float = 4,
    ) -> float:
        width = CONTENT_WIDTH - 20
        for item in items:
            self.canvas.setFillColor(BLUE)
            self.canvas.circle(MARGIN_X + 4, y + 3, 2.1, fill=1, stroke=0)
            y = self.text(
                item,
                y,
                x=MARGIN_X + 16,
                width=width,
                size=size,
                leading=leading,
            )
            y -= gap
        return y

    def callout(
        self,
        title: str,
        body: str,
        y: float,
        *,
        fill: colors.Color = PALE_BLUE,
        title_color: colors.Color = BLUE_DARK,
    ) -> float:
        inner_width = CONTENT_WIDTH - 28
        body_lines = wrap_text(body, "GuideSans", 9.2, inner_width)
        height = 39 + max(1, len(body_lines)) * 13.5
        self.canvas.setFillColor(fill)
        self.canvas.setStrokeColor(LINE)
        self.canvas.roundRect(MARGIN_X, y - height, CONTENT_WIDTH, height, 6, fill=1, stroke=1)
        self.text(
            title,
            y - 20,
            x=MARGIN_X + 14,
            width=inner_width,
            size=10.2,
            bold=True,
            color=title_color,
        )
        self.text(
            body,
            y - 38,
            x=MARGIN_X + 14,
            width=inner_width,
            size=9.2,
            leading=13.5,
        )
        return y - height - 12

    def screenshot(
        self,
        name: str,
        y: float,
        *,
        max_height: float = 320,
        caption: str = SCREENSHOT_CAPTION,
    ) -> float:
        image_path = self.assets / name
        if not image_path.is_file():
            raise FileNotFoundError(f"Screenshot is missing: {image_path}")
        with Image.open(image_path) as image:
            source_width, source_height = image.size
        width = CONTENT_WIDTH
        height = width * source_height / source_width
        if height > max_height:
            height = max_height
            width = height * source_width / source_height
        x = (PAGE_WIDTH - width) / 2
        self.canvas.setFillColor(WHITE)
        self.canvas.setStrokeColor(LINE)
        self.canvas.roundRect(x - 3, y - height - 3, width + 6, height + 6, 5, fill=1, stroke=1)
        self.canvas.drawImage(
            ImageReader(str(image_path)),
            x,
            y - height,
            width=width,
            height=height,
            preserveAspectRatio=True,
            mask="auto",
        )
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("GuideSans", 8)
        self.canvas.drawRightString(PAGE_WIDTH - MARGIN_X, y - height - 15, caption)
        return y - height - 29

    def numbered_steps(
        self,
        items: list[tuple[str, str]],
        y: float,
        *,
        height: float = 82,
    ) -> float:
        gap = 8
        width = (CONTENT_WIDTH - gap * (len(items) - 1)) / len(items)
        for index, (title, body) in enumerate(items, 1):
            x = MARGIN_X + (index - 1) * (width + gap)
            self.canvas.setFillColor(WHITE)
            self.canvas.setStrokeColor(LINE)
            self.canvas.roundRect(x, y - height, width, height, 6, fill=1, stroke=1)
            self.canvas.setFillColor(BLUE)
            self.canvas.circle(x + 18, y - 19, 10, fill=1, stroke=0)
            self.canvas.setFillColor(WHITE)
            self.canvas.setFont("GuideSans-Bold", 9)
            self.canvas.drawCentredString(x + 18, y - 22, str(index))
            self.text(
                title,
                y - 22,
                x=x + 34,
                width=width - 43,
                size=9.5,
                bold=True,
            )
            self.text(
                body,
                y - 45,
                x=x + 12,
                width=width - 24,
                size=8.5,
                leading=12,
                color=MUTED,
            )
        return y - height - 12

    def table(
        self,
        rows: list[tuple[str, str]],
        y: float,
        *,
        label_width: float = 112,
        min_height: float = 43,
    ) -> float:
        for index, (label, value) in enumerate(rows):
            value_width = CONTENT_WIDTH - label_width - 22
            value_lines = wrap_text(value, "GuideSans", 8.9, value_width)
            label_lines = wrap_text(label, "GuideSans-Bold", 9.1, label_width - 20)
            height = max(
                min_height,
                18 + max(len(value_lines), len(label_lines)) * 12.5,
            )
            self.canvas.setFillColor(PANEL if index % 2 == 0 else WHITE)
            self.canvas.setStrokeColor(LINE)
            self.canvas.rect(MARGIN_X, y - height, CONTENT_WIDTH, height, fill=1, stroke=1)
            self.text(
                label,
                y - 18,
                x=MARGIN_X + 10,
                width=label_width - 20,
                size=9.1,
                leading=12.5,
                bold=True,
                color=BLUE_DARK,
            )
            self.text(
                value,
                y - 18,
                x=MARGIN_X + label_width,
                width=value_width,
                size=8.9,
                leading=12.5,
            )
            y -= height
        return y - 12

    def mini_table(
        self,
        title: str,
        rows: list[tuple[str, str]],
        *,
        x: float,
        y: float,
        width: float,
    ) -> float:
        self.canvas.setFillColor(INK)
        self.canvas.setFont("GuideSans-Bold", 11.5)
        self.canvas.drawString(x, y, title)
        y -= 18
        label_width = width * 0.31
        for index, (label, value) in enumerate(rows):
            value_width = width - label_width - 18
            lines = max(
                len(wrap_text(label, "GuideSans-Bold", 8.4, label_width - 14)),
                len(wrap_text(value, "GuideSans", 8.3, value_width)),
            )
            height = max(38, 15 + lines * 11.5)
            self.canvas.setFillColor(PANEL if index % 2 == 0 else WHITE)
            self.canvas.setStrokeColor(LINE)
            self.canvas.rect(x, y - height, width, height, fill=1, stroke=1)
            self.text(
                label,
                y - 15,
                x=x + 8,
                width=label_width - 14,
                size=8.4,
                leading=11.5,
                bold=True,
                color=BLUE_DARK,
            )
            self.text(
                value,
                y - 15,
                x=x + label_width,
                width=value_width,
                size=8.3,
                leading=11.5,
            )
            y -= height
        return y

    def cover(self) -> None:
        self.page_number = 1
        self.canvas.setFillColor(colors.HexColor("#F4F7FB"))
        self.canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)

        self.canvas.setFillColor(BLUE)
        self.canvas.roundRect(MARGIN_X, PAGE_HEIGHT - 104, 42, 42, 9, fill=1, stroke=0)
        self.canvas.setFillColor(WHITE)
        self.canvas.setFont("GuideSans-Bold", 18)
        self.canvas.drawCentredString(MARGIN_X + 21, PAGE_HEIGHT - 90, "搜")

        self.canvas.setFillColor(BLUE_DARK)
        self.canvas.setFont("GuideSans-Bold", 25)
        self.canvas.drawString(MARGIN_X + 56, PAGE_HEIGHT - 82, "本地多格式全文搜索工具")
        self.canvas.setFont("GuideSans-Bold", 16)
        self.canvas.drawString(MARGIN_X + 56, PAGE_HEIGHT - 107, "图文使用说明")
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("GuideSans", 9.5)
        self.canvas.drawRightString(
            PAGE_WIDTH - MARGIN_X,
            PAGE_HEIGHT - 105,
            f"版本 {VERSION} · {GUIDE_DATE}",
        )

        y = PAGE_HEIGHT - 145
        y = self.screenshot(
            "search_results.png",
            y,
            max_height=306,
            caption=SCREENSHOT_CAPTION,
        )
        y = self.numbered_steps(
            [
                ("添加范围", "在“索引管理”中选择要搜索的本地文件夹。"),
                ("完成索引", "全部可解析文件完成后，搜索才会开放。"),
                ("输入关键词", "在结果和右侧预览中查看命中与黄显。"),
            ],
            y,
            height=78,
        )
        self.callout(
            "完全本地运行",
            "索引、OCR 与搜索均在本机完成，不需要联网；视频文件不解析，也不会计入解析失败。",
            y,
            fill=PALE_GREEN,
        )
        self.footer()

    def build(self) -> None:
        self.cover()

        y = self.begin_page(
            "01  首次建立索引",
            "先添加搜索范围，再等待全部可解析文件完成；首轮完成前搜索保持关闭",
        )
        y = self.screenshot("index_running.png", y, max_height=305)
        y = self.numbered_steps(
            [
                ("添加搜索范围", "点击右上角按钮，选择一个或多个本地文件夹。"),
                ("选择运行方式", "办公时用普通模式；电脑空闲时用性能模式。"),
                ("等待完整完成", "总数、失败数、排除视频和当前阶段会持续更新。"),
            ],
            y,
            height=78,
        )
        self.callout(
            "首次索引为什么不能立即搜索？",
            "本工具要求当前范围内全部可解析文件都完成后才开放搜索，避免用户误以为没有结果。视频会明确排除；损坏、加密或缺少转换组件的文件会进入“未成功索引”。",
            y,
            fill=PALE_AMBER,
        )
        self.footer()

        y = self.begin_page(
            "02  索引状态、安全暂停与模式切换",
            "预计时间只显示一个值；真正进入“已暂停”后才允许切换普通或性能模式",
        )
        y = self.screenshot("index_paused.png", y, max_height=300)
        y = self.table(
            [
                ("1. 点击暂停", "界面先显示“正在暂停…”，程序立即停止提交新文件。"),
                ("2. 等待已暂停", "活动任务完成当前页、OCR 区域或安全检查点后停止；此时 CPU 和磁盘活动应接近零。"),
                ("3. 切换模式", "只有显示“已暂停”时，普通模式和性能模式按钮才可用；切换不会重复解析已完成文件。"),
                ("4. 点击继续", "剩余任务从安全检查点继续，并按新模式重建所需进程池；预计时间重新估算。"),
            ],
            y,
            label_width=104,
            min_height=42,
        )
        self.callout(
            "预计时间与卡滞判断",
            "样本不足时显示“正在估算…”，稳定后显示“预计剩余约 N 分钟”，暂停时停止变化。长文件没有固定总耗时上限；只要页、区域或 ZIP 成员仍有有效进展就会继续，长期无进展才会回收并重试。",
            y,
        )
        self.footer()

        y = self.begin_page(
            "03  搜索、黄显与打开命中位置",
            "中间结果用于选择文件，右侧预览展示完整命中上下文并同步黄显关键词",
        )
        y = self.screenshot("search_results.png", y, max_height=305)
        y = self.bullets(
            [
                "输入文件名、路径或正文关键词；中文查询会兼容常见空格差异，例如“拔掉 3 个传感器”可命中“拔掉3 个传感器”。",
                "中间结果卡显示格式、文件名、命中位置、路径、大小和修改时间；同一文件多处命中时可展开查看。",
                "右侧“命中上下文”显示当前文件的命中片段，关键词使用黄色背景突出显示；内容较长时在预览区滚动查看。",
                "点击蓝色位置文本可尝试定位到 PDF 页、幻灯片、工作表/单元格或文本行；底部按钮用于普通打开、打开所在文件夹和复制路径。",
            ],
            y,
            gap=3,
        )
        self.callout(
            "定位失败不等于搜索失败",
            "不同 PDF 阅读器、Microsoft Office 或 WPS 版本的定位能力不同。程序仍会打开原文件，并明确提示是否确认定位成功；不会把普通打开冒充为定位成功。",
            y,
            fill=PALE_BLUE,
        )
        self.footer()

        y = self.begin_page(
            "04  匹配方式与支持格式",
            "需要缩小结果时，优先增加关键词、限定格式或缩小搜索范围",
        )
        y = self.screenshot("search_controls.png", y, max_height=135)
        table_y = y - 2
        left_x = MARGIN_X
        column_gap = 10
        column_width = (CONTENT_WIDTH - column_gap) / 2
        left_bottom = self.mini_table(
            "匹配方式",
            [
                ("完全包含", "按输入内容直接匹配，适合普通查询。"),
                ("完整短语", "要求词序连续，适合固定术语。"),
                ("全部关键词", "每个关键词都必须出现。"),
                ("任意关键词", "出现任一关键词即可。"),
                ("正则表达式", "适合明确规则；可能需要扫描更多记录。"),
            ],
            x=left_x,
            y=table_y,
            width=column_width,
        )
        right_bottom = self.mini_table(
            "索引格式",
            [
                ("PDF", "原生文字直接提取；扫描页按需 OCR。"),
                ("Word", "docx 直接解析；doc 需本机 Office/WPS 等软件转换。"),
                ("Excel", ".xlsx/.xlsm；大型工作簿分段读取；.xls 走转换车道。"),
                ("PowerPoint", ".pptx；.ppt 走本机转换车道。"),
                ("图片", "JPG/PNG/BMP/TIFF，采用 960 检测、原图区域识别和低质量分块。"),
                ("文本/ZIP", "常见文本直接读取；ZIP 在安全限制内递归解析受支持成员。"),
            ],
            x=left_x + column_width + column_gap,
            y=table_y,
            width=column_width,
        )
        y = min(left_bottom, right_bottom) - 12
        self.callout(
            "视频文件",
            "视频不在解析范围内。界面会显示“排除视频 N”，不会把视频算成索引失败，也不会尝试提取画面、字幕或音轨。",
            y,
            fill=PALE_AMBER,
        )
        self.footer()

        y = self.begin_page(
            "05  未成功索引与保底开放",
            "先正常重试；确认文件无法修复或无需搜索后，再排除或使用最终保底流程",
        )
        y = self.screenshot("failed_files.png", y, max_height=300)
        y = self.numbered_steps(
            [
                ("重新尝试", "文件已修复、解密或转换软件恢复时使用；不会排除文件。"),
                ("从范围排除", "确认文件不需要搜索时使用；只改索引，源文件不会删除。"),
                ("保底完成", "先做最后一次重试；仍失败时再次确认排除并开放其他文件。"),
            ],
            y,
            height=86,
        )
        self.callout(
            "保底操作有两次确认",
            "第一次只确认最后一次正常重试，不会立即排除文件；仍失败后第二次才确认排除。若阻断项为 0 但仍有残留任务或未发布全文索引，保底按钮也可修复状态，不会新增人工排除记录。",
            y,
            fill=PALE_RED,
        )
        self.footer()

        y = self.begin_page(
            "06  设置与数据库迁移",
            "索引数据库默认位于本机用户目录；C 盘空间不足时可迁移到本机固定磁盘",
        )
        y = self.screenshot("database_settings_detail.png", y, max_height=215)
        y = self.table(
            [
                ("1. 选择目录", "在“设置 > 索引数据库”中选择本机固定磁盘上的目标目录；网络共享、程序安装目录和不可写目录会被拒绝。"),
                ("2. 确认空间", "界面显示当前数据库大小和目标磁盘可用空间；迁移前请停止搜索、索引和失败文件处理任务。"),
                ("3. 开始迁移", "程序依次执行准备、传输、完整性校验、切换路径和重新打开；失败或取消会继续使用原数据库。"),
                ("4. 核对完成", "迁移成功后无需重新索引。原数据库保留为带时间戳的回滚备份，当前路径写入用户级位置配置。"),
            ],
            y,
            label_width=102,
            min_height=45,
        )
        self.callout(
            "升级与重复启动",
            "升级时把完整 ZIP 解压到新目录，不要只覆盖 EXE，也不要混用不同版本的 _internal。新版本会读取已迁移的数据库位置。程序采用单实例运行，重复启动会激活已有窗口，不会再创建一个写库进程。",
            y,
            fill=PALE_GREEN,
        )
        self.footer()

        y = self.begin_page(
            "07  日常更新、故障排查与隐私",
            "增量更新不会重复解析未变化文件；遇到异常时先保留数据库和日志",
        )
        y = self.screenshot("index_status.png", y, max_height=95)
        y = self.heading("日常使用", y)
        y = self.bullets(
            [
                "点击“更新”或“更新全部”只处理新增、修改和失效文件；内容未变化时会复用已有解析结果与缓存。",
                "普通模式适合边办公边更新；性能模式适合电脑空闲时的首次全量索引。首次运行 OCR 可能需要加载本地模型，后续任务复用常驻进程。",
                "取消会结束本轮更新并按事务边界回滚未发布内容；安全暂停用于稍后继续，两者含义不同。",
            ],
            y,
        )
        y = self.heading("最短排查顺序", y)
        y = self.table(
            [
                ("索引像是卡住", "观察完成数、当前阶段、页/区域进度和“距上次进展”；长期无有效进展时程序会自动回收并重试。"),
                ("文件搜不到", "确认索引已经开放、查询范围/格式正确，再到“未成功索引”检查该文件是否失败或被人工排除。"),
                ("程序点了没反应", "确认完整解压且 _internal 与 EXE 同目录；查看任务栏中是否已有窗口，并收集启动日志。"),
                ("需要提供日志", "复制 %LOCALAPPDATA%\\LocalFullTextSearch\\logs，重点包括 startup.log、startup-state.json 和 app.log。"),
            ],
            y,
            label_width=110,
            min_height=46,
        )
        self.callout(
            "隐私与数据位置",
            "文件正文、OCR、索引和搜索记录均不上传。\n默认数据目录：%LOCALAPPDATA%\\LocalFullTextSearch。数据库、设置、日志和 OCR 缓存均保存在这里；不要为解决单个失败文件直接删除整个目录。",
            y,
            fill=PALE_GREEN,
        )
        self.footer()

        if self.page_number != PAGE_TOTAL:
            raise RuntimeError(
                f"Guide page count mismatch: built={self.page_number}; expected={PAGE_TOTAL}"
            )
        self.canvas.save()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the illustrated release user guide PDF")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path.cwd() / "docs" / "assets" / "user-guide",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    register_fonts()
    Guide(args.output.resolve(), args.assets.resolve()).build()
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
