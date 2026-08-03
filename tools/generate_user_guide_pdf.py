from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader


PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 48
TOP_Y = PAGE_HEIGHT - 54
BOTTOM_Y = 42
BLUE = colors.HexColor("#2563EB")
DEEP_BLUE = colors.HexColor("#153E75")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#596579")
PALE_BLUE = colors.HexColor("#EFF6FF")
PALE_GREEN = colors.HexColor("#ECFDF5")
PALE_AMBER = colors.HexColor("#FFFBEB")
BORDER = colors.HexColor("#D8E0EA")
PANEL = colors.HexColor("#F7F9FC")
VERSION = "1.3.7"
GUIDE_DATE = "2026 年 8 月 3 日"
PAGE_TOTAL = 10


def register_fonts() -> None:
    regular = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("Microsoft YaHei fonts are required to build the guide")
    pdfmetrics.registerFont(TTFont("GuideSans", str(regular)))
    pdfmetrics.registerFont(TTFont("GuideSans-Bold", str(bold)))


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
        self.page_number = 0

    def new_page(self, title: str, subtitle: str = "") -> float:
        if self.page_number:
            self.canvas.showPage()
        self.page_number += 1
        self.canvas.setFillColor(DEEP_BLUE)
        self.canvas.setFont("GuideSans-Bold", 21)
        self.canvas.drawString(MARGIN_X, TOP_Y, title)
        if subtitle:
            self.canvas.setFillColor(MUTED)
            self.canvas.setFont("GuideSans", 9.5)
            self.canvas.drawString(MARGIN_X, TOP_Y - 20, subtitle)
        self.canvas.setStrokeColor(BORDER)
        self.canvas.line(MARGIN_X, TOP_Y - 31, PAGE_WIDTH - MARGIN_X, TOP_Y - 31)
        return TOP_Y - 55

    def footer(self) -> None:
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("GuideSans", 8)
        self.canvas.drawString(MARGIN_X, 24, f"本地多格式全文搜索工具 · 使用说明 {VERSION}")
        self.canvas.drawRightString(
            PAGE_WIDTH - MARGIN_X, 24, f"{self.page_number} / {PAGE_TOTAL}"
        )

    def heading(self, text: str, y: float) -> float:
        self.canvas.setFillColor(INK)
        self.canvas.setFont("GuideSans-Bold", 13)
        self.canvas.drawString(MARGIN_X, y, text)
        return y - 22

    def paragraph(
        self,
        text: str,
        y: float,
        *,
        x: float = MARGIN_X,
        width: float | None = None,
        size: float = 10,
        leading: float = 16,
        color: colors.Color = INK,
        bold: bool = False,
    ) -> float:
        width = width or PAGE_WIDTH - 2 * MARGIN_X
        font = "GuideSans-Bold" if bold else "GuideSans"
        self.canvas.setFillColor(color)
        self.canvas.setFont(font, size)
        for line in wrap_text(text, font, size, width):
            self.canvas.drawString(x, y, line)
            y -= leading
        return y

    def bullets(self, items: list[str], y: float, *, width: float | None = None) -> float:
        width = width or PAGE_WIDTH - 2 * MARGIN_X - 20
        for item in items:
            self.canvas.setFillColor(BLUE)
            self.canvas.circle(MARGIN_X + 4, y + 3, 2.2, fill=1, stroke=0)
            y = self.paragraph(
                item,
                y,
                x=MARGIN_X + 16,
                width=width,
                size=9.5,
                leading=15,
            )
            y -= 5
        return y

    def callout(
        self,
        title: str,
        body: str,
        y: float,
        *,
        fill: colors.Color = PALE_BLUE,
        height: float = 70,
    ) -> float:
        x = MARGIN_X
        width = PAGE_WIDTH - 2 * MARGIN_X
        self.canvas.setFillColor(fill)
        self.canvas.setStrokeColor(BORDER)
        self.canvas.roundRect(x, y - height, width, height, 8, fill=1, stroke=1)
        self.canvas.setFillColor(DEEP_BLUE)
        self.canvas.setFont("GuideSans-Bold", 10.5)
        self.canvas.drawString(x + 14, y - 22, title)
        self.paragraph(
            body,
            y - 41,
            x=x + 14,
            width=width - 28,
            size=9,
            leading=14,
            color=INK,
        )
        return y - height - 14

    def screenshot(self, name: str, y: float, *, max_height: float = 320) -> float:
        image_path = self.assets / name
        if not image_path.exists():
            raise FileNotFoundError(f"Screenshot is missing: {image_path}")
        with Image.open(image_path) as image:
            source_width, source_height = image.size
        width = PAGE_WIDTH - 2 * MARGIN_X
        height = width * source_height / source_width
        if height > max_height:
            height = max_height
            width = height * source_width / source_height
        x = (PAGE_WIDTH - width) / 2
        self.canvas.setFillColor(colors.white)
        self.canvas.setStrokeColor(BORDER)
        self.canvas.roundRect(x - 4, y - height - 4, width + 8, height + 8, 6, fill=1, stroke=1)
        self.canvas.drawImage(
            ImageReader(str(image_path)),
            x,
            y - height,
            width=width,
            height=height,
            preserveAspectRatio=True,
            mask="auto",
        )
        return y - height - 18

    def table(
        self,
        rows: list[tuple[str, str]],
        y: float,
        *,
        first_width: float = 120,
        row_height: float = 38,
    ) -> float:
        x = MARGIN_X
        width = PAGE_WIDTH - 2 * MARGIN_X
        for index, (label, value) in enumerate(rows):
            fill = PANEL if index % 2 == 0 else colors.white
            self.canvas.setFillColor(fill)
            self.canvas.setStrokeColor(BORDER)
            self.canvas.rect(x, y - row_height, width, row_height, fill=1, stroke=1)
            self.canvas.setFillColor(DEEP_BLUE)
            self.canvas.setFont("GuideSans-Bold", 9.5)
            self.canvas.drawString(x + 12, y - 23, label)
            self.paragraph(
                value,
                y - 15,
                x=x + first_width,
                width=width - first_width - 12,
                size=9,
                leading=13,
            )
            y -= row_height
        return y - 26

    def cover(self) -> None:
        self.page_number = 1
        self.canvas.setFillColor(colors.HexColor("#F2F7FF"))
        self.canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        self.canvas.setFillColor(BLUE)
        self.canvas.roundRect(MARGIN_X, PAGE_HEIGHT - 175, 54, 54, 13, fill=1, stroke=0)
        self.canvas.setFillColor(colors.white)
        self.canvas.setFont("GuideSans-Bold", 24)
        self.canvas.drawCentredString(MARGIN_X + 27, PAGE_HEIGHT - 158, "搜")
        self.canvas.setFillColor(DEEP_BLUE)
        self.canvas.setFont("GuideSans-Bold", 28)
        self.canvas.drawString(MARGIN_X, PAGE_HEIGHT - 245, "本地多格式全文搜索工具")
        self.canvas.setFont("GuideSans-Bold", 20)
        self.canvas.drawString(MARGIN_X, PAGE_HEIGHT - 282, "使用说明")
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("GuideSans", 11)
        self.canvas.drawString(MARGIN_X, PAGE_HEIGHT - 315, f"适用于 {VERSION} · {GUIDE_DATE}")
        self.canvas.setFillColor(colors.white)
        self.canvas.setStrokeColor(BORDER)
        self.canvas.roundRect(
            MARGIN_X,
            PAGE_HEIGHT - 510,
            PAGE_WIDTH - 2 * MARGIN_X,
            128,
            12,
            fill=1,
            stroke=1,
        )
        self.canvas.setFillColor(INK)
        self.canvas.setFont("GuideSans-Bold", 12)
        self.canvas.drawString(MARGIN_X + 20, PAGE_HEIGHT - 414, "本版本重点")
        self.bullets(
            [
                "首次索引完成前不可搜索，避免把未索引文件误判为“搜不到”。",
                "人工排除在后台更新全文索引；窗口保持响应并持续显示当前阶段。",
                "“强力完成本次索引”固定显示，可修复阻断文件或零阻断残留状态。",
                "性能模式按当前硬件和空闲资源提高 PDF、OCR 与普通文件并发。",
            ],
            PAGE_HEIGHT - 440,
            width=PAGE_WIDTH - 2 * MARGIN_X - 48,
        )
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("GuideSans", 9)
        self.canvas.drawString(MARGIN_X, 52, "离线运行 · 索引数据库保存在本机 · 视频文件不参与解析")
        self.footer()

    def build(self) -> None:
        self.cover()

        y = self.new_page("01  首次索引", "等待全部可解析文件处理完成后，搜索功能才开放")
        y = self.screenshot("index_ui_validation.png", y, max_height=300)
        y = self.heading("状态条怎么读", y)
        y = self.bullets(
            [
                "第一行显示模式、完成数/总数、失败数、排除视频数、单一预计剩余时间和控制按钮。",
                "第二行显示当前文件、OCR 已运行时间、中文阶段说明、进度单位及距上次有效进展时间。",
                "样本不足时显示“正在估算…”。估算稳定后显示“预计剩余约 11 分钟”，不会再显示跨度很大的时间范围。",
                "路径太长时界面只缩短文件名；鼠标悬停仍可查看完整路径。",
            ],
            y,
        )
        self.callout(
            "为什么索引中不能搜索？",
            "本工具采用“完整索引后开放搜索”的规则。视频和仅元数据完成项不属于失败；确实无法解析的文件可在“未成功索引”中人工排除，排除最后一个阻断项后会自动发布索引并开放搜索。",
            y - 2,
            fill=PALE_AMBER,
            height=76,
        )
        self.footer()

        y = self.new_page("02  安全暂停与模式切换", "暂停不是强杀进程：先到安全检查点，再确认“已暂停”")
        y = self.screenshot("index_paused_ui_validation.png", y, max_height=300)
        y = self.heading("正确操作顺序", y)
        y = self.table(
            [
                ("1. 点击暂停", "界面先显示“正在暂停…”，立即停止提交新文件。"),
                ("2. 等待已暂停", "活动任务完成当前页、OCR 区域或安全检查点；此时 CPU、磁盘活动应接近零。"),
                ("3. 切换模式", "只有“已暂停”时，普通模式与性能模式按钮才可用。"),
                ("4. 点击继续", "已完成文件不会重复解析，剩余任务按新模式运行；预计时间重新估算。"),
            ],
            y,
            first_width=112,
            row_height=44,
        )
        self.callout(
            "暂停与取消的区别",
            "暂停用于稍后继续，不破坏缓存和数据库；取消用于结束本轮索引。需要释放资源时优先等待“已暂停”，不要直接结束进程。",
            y,
            fill=PALE_GREEN,
            height=70,
        )
        self.footer()

        y = self.new_page("03  搜索与命中高亮", "正文、文件名、表格、幻灯片和图片 OCR 文字统一检索")
        y = self.screenshot("ui_validation.png", y, max_height=305)
        y = self.heading("搜索结果说明", y)
        y = self.bullets(
            [
                "中间结果卡片和右侧“命中上下文”都会用黄色标出搜索内容。",
                "空格差异可容忍，例如“拔掉3 个传感器”可由“拔掉 3 个传感器”命中。",
                "结果卡会完整保留正文片段、路径和日期；内容较长时可在结果区滚动查看。",
                "搜索框右侧按钮保留固定空间，不会覆盖输入文字；可按格式、范围和匹配方式筛选。",
                "普通名称和正文查询优先使用本地 FTS 倒排索引，不重新读取源文件；文件名结果先显示。",
                "慢搜索显示阶段、已找到数量和耗时；扫描型搜索显示真实百分比，并可点击停止。",
                "双击结果或使用右侧按钮打开原文件；也可打开所在文件夹、复制完整路径。",
            ],
            y,
        )
        self.footer()

        y = self.new_page("04  支持格式与处理方式", "不同格式进入独立解析车道，避免一个慢文件拖住其他类型")
        y = self.table(
            [
                ("PDF", "先提取可复制正文；扫描页或图纸按需进入 OCR。PDF 使用独立进程车道。"),
                ("Word", "DOCX 直接解析；DOC 通过本机可用的 Office/兼容转换组件转为临时 DOCX 后解析。"),
                ("Excel", "XLSX/XLSM 使用流式读取，大表按工作表分段；XLS 走旧版 Office 专用车道。"),
                ("PowerPoint", "PPTX 提取幻灯片及备注文字；PPT 走旧版 Office 专用车道。"),
                ("图片", "JPG、PNG、BMP、TIFF 等进入自适应 OCR，索引识别出的文字。"),
                ("压缩包", "ZIP 成员按内部格式分派；受成员数量、解压总量和嵌套层级限制。"),
                ("视频", "不解析、不 OCR，不计入失败；状态条单独显示“排除视频”。"),
            ],
            y,
            first_width=100,
            row_height=48,
        )
        y = self.callout(
            "旧版 Office 的前提",
            "DOC、XLS、PPT 的转换能力取决于电脑上可调用的 Office 或兼容组件。转换结果会缓存，文件未变化时不会重复转换。",
            y,
            fill=PALE_AMBER,
            height=72,
        )
        self.footer()

        y = self.new_page("05  图片与 PDF OCR", "960 检测 + 原图文字区域识别 + 低质量时原图分块")
        y = self.heading("图片 OCR 的三步策略", y)
        y = self.table(
            [
                ("快速检测", "把整图限制到最长边 960 像素，仅用于寻找可能的文字区域。"),
                ("原图识别", "识别时回到原始像素裁剪文字区域，不用 960 像素缩略图识别正文。"),
                ("质量兜底", "区域太少、置信度偏低或小字密集时，对原图分块并批量识别，优先保证召回。"),
            ],
            y,
            first_width=112,
            row_height=60,
        )
        y = self.heading("PDF 动态分辨率", y)
        y = self.bullets(
            [
                "150 DPI 用于低成本预检；普通文字区域以 200 DPI 识别。",
                "只有低置信度或小字号区域升级到 300 DPI，避免整页无差别高分辨率渲染。",
                "大型图纸若区域结果过少，会自动回退到完整分块识别；因此个别极端页面仍可能耗时较长。",
                "OCR 模型在进程中常驻，同一批任务复用模型；模型或算法指纹变化时自动失效旧缓存。",
            ],
            y,
        )
        self.callout(
            "精度原则",
            "缩放图只负责“找文字”，最终文字识别使用原图区域。保底分块不会为了速度跳过不确定区域，因此首次索引可能较慢，但可降低漏字风险。",
            y,
            fill=PALE_GREEN,
            height=76,
        )
        self.footer()

        y = self.new_page("06  长文件、卡滞与失败", "超时依据“是否有有效进展”，不是限制文件总耗时")
        y = self.heading("无进展超时如何工作", y)
        y = self.table(
            [
                ("有效进展", "页码、工作表、ZIP 成员、OCR 区域或输出块等语义游标发生前进。"),
                ("不算进展", "只重复发送心跳、阶段和游标均未变化，不会重置超时计时。"),
                ("到达阈值", "回收对应解析进程并从安全检查点重试，其他文件和界面不被永久卡住。"),
                ("重复卡点", "同一阶段、同一游标连续失败后进入“未成功索引”，保留诊断信息。"),
            ],
            y,
            first_width=112,
            row_height=56,
        )
        y = self.heading("遇到失败时", y)
        y = self.bullets(
            [
                "在“未成功索引”页面查看文件、失败分类和原因；先确认原文件是否能正常打开。",
                "密码保护、文件损坏、缺少旧版 Office 转换组件或 ZIP 超过安全限制都可能失败。",
                "修复原文件或环境后执行更新；已完成且未变化的文件会跳过，不必重建全部内容。",
                "不要删除本地索引数据库来解决单文件问题，除非需要彻底重建或数据库已损坏。",
            ],
            y,
        )
        self.footer()

        y = self.new_page("07  未成功索引与保底完成", "优先正常恢复；确认无法解析后再排除，源文件始终保留")
        y = self.heading("三个控件怎么选", y)
        y = self.table(
            [
                ("重新尝试", "先用当前文件和环境再处理一次。原文件修复、解密或补齐转换组件后优先使用。"),
                ("从当前范围排除", "确认后立即转入后台，显示审计、清理、重建和状态刷新阶段；排除最后一个阻断项后自动开放搜索。"),
                ("强力完成本次索引", "用于最终保底。有阻断时先正常重试，仍失败才确认排除；零阻断残留状态则直接修复并发布索引。"),
            ],
            y,
            first_width=132,
            row_height=66,
        )
        y = self.heading("“强力完成本次索引”按钮状态", y)
        y = self.bullets(
            [
                "按钮固定显示在“未成功索引”页面顶部操作栏，不需要滚动查找。",
                "人工排除期间显示忙碌进度和大型 FTS 状态；搜索、索引、恢复和强力完成暂时禁用。",
                "索引真正就绪，或索引/后台更新任务仍在运行：按钮为灰色禁用。",
                "存在阻断项，或零阻断但有残留任务、未发布候选、未收敛批次、待发布 FTS：按钮为蓝色可用。",
                "人工排除和强力完成都不会删除原文件；被排除文件的文件名、路径和正文不参与搜索。",
                "点击“取消索引更新”会回滚本次事务；原文件始终保留，旧排除记录会在源文件变化后失效。",
            ],
            y,
        )
        y = self.callout(
            "二次确认保护",
            "强力完成不会静默跳过文件。第一次确认只执行最后一次正常恢复；仍有阻断项时会再次列出文件，只有用户第二次明确确认后才排除并开放搜索。",
            y,
            fill=PALE_AMBER,
            height=76,
        )
        self.footer()

        y = self.new_page("08  性能模式与异常演示", "电脑空闲时提高资源利用；演示入口与正式索引完全隔离")
        y = self.heading("普通模式和性能模式", y)
        y = self.table(
            [
                ("普通模式", "适合边索引边办公，CPU、磁盘与内存占用更克制。"),
                ("性能模式", "适合电脑空闲时首次全量索引；根据 CPU、可用内存和磁盘情况动态提高各解析车道并发。"),
                ("切换方法", "运行中先点击暂停，确认界面显示“已暂停”后切换，再点击继续。"),
                ("时间判断", "预计时间是动态参考；更应关注完成数、当前阶段和进度是否持续前进。"),
            ],
            y,
            first_width=112,
            row_height=53,
        )
        y = self.heading("亲自验证搜索保底", y)
        y = self.bullets(
            [
                "关闭正式程序，双击发行目录根部的“异常文件保底功能演示.bat”。",
                "演示会重置包内 failure-fallback-demo-data，模拟损坏 PDF、密码 Word、失败图片和超限 ZIP。",
                "在“未成功索引”中亲自测试重新尝试、从当前索引范围排除和强力完成。",
                "开放搜索后输入 DEMO_SEARCH_OK，应命中正常演示文件。再次运行批处理可重置演示。",
            ],
            y,
        )
        self.callout(
            "不会影响正式数据库",
            "演示入口使用独立数据目录，不读取或修改 %LOCALAPPDATA%\\LocalFullTextSearch\\data\\search_index.db。演示中的排除和搜索结果不会写入正式索引。",
            y,
            fill=PALE_GREEN,
            height=76,
        )
        self.footer()

        y = self.new_page("09  升级、数据库与启动日志", "整包升级可继续使用原索引；点 EXE 没反应时先收集启动日志")
        y = self.heading("升级版本的正确方式", y)
        y = self.table(
            [
                ("1. 退出旧版", "确认旧程序已经关闭，保留当前数据库和日志。"),
                ("2. 解压整包", "把新版 ZIP 解压到完整的新目录，不要只把新 EXE 覆盖进旧 _internal。"),
                ("3. 启动新版", "新版自动复用 %LOCALAPPDATA%\\LocalFullTextSearch\\data\\search_index.db，并按需迁移。"),
                ("4. 增量更新", "解析规则更新后可执行“更新全部”；未变化且仍有效的内容会尽量复用。"),
            ],
            y,
            first_width=112,
            row_height=54,
        )
        y = self.heading("程序点了没反应时", y)
        y = self.bullets(
            [
                "优先复制整个 %LOCALAPPDATA%\\LocalFullTextSearch\\logs 目录。",
                "重点文件：startup.log、startup-state.json、app.log 以及 app.log.*。",
                "本地数据目录不可写时，再检查 %TEMP%\\LocalFullTextSearch。",
                "若被 Windows 安全提示、EDR 或 DLL 加载阶段拦截，可能来不及生成日志；同时提供完整目录结构、提示截图和安全软件记录。",
            ],
            y,
        )
        y = self.callout(
            "不要轻易删除数据库",
            "单个文件失败不需要删除数据库。只有明确确认数据库损坏或需要彻底重建时才清空数据目录；回滚旧版时不要混用不同版本的 _internal。",
            y,
            fill=PALE_AMBER,
            height=76,
        )
        y = self.heading("交付检查", y)
        self.bullets(
            [
                "完整目录应包含 EXE、_internal、说明书、异常演示入口和“发行资料”。",
                "保留发行资料中的 BUILD-INFO、SHA256SUMS 和验证结果，便于核对版本。",
            ],
            y,
        )
        self.footer()

        self.canvas.save()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the release user guide PDF")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path.cwd())
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    register_fonts()
    Guide(args.output.resolve(), args.assets.resolve()).build()
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
