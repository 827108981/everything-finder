from __future__ import annotations

from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.errors import UnsupportedFormatError
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.docx_parser import DocxParser
from local_full_text_search.parsers.image_parser import ImageParser
from local_full_text_search.parsers.legacy_office_parser import LegacyOfficeParser
from local_full_text_search.parsers.metadata_parser import MetadataOnlyParser
from local_full_text_search.parsers.pdf_parser import PdfParser
from local_full_text_search.parsers.pptx_parser import PptxParser
from local_full_text_search.parsers.text_parser import TextParser
from local_full_text_search.parsers.xlsx_parser import XlsxParser
from local_full_text_search.parsers.zip_parser import ZipParser


class ParserRegistry:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        parsers: list[BaseParser] = [
            TextParser(),
            PdfParser(
                enable_scanned_ocr=settings.enable_ocr and settings.ocr_scanned_pdf,
                ocr_language=settings.ocr_language,
            ),
            DocxParser(),
            XlsxParser(),
            PptxParser(),
            ImageParser(
                language=settings.ocr_language,
                enabled=settings.enable_ocr and settings.ocr_images,
                min_pixels=settings.min_ocr_image_pixels,
                max_side=settings.max_ocr_image_side,
            ),
            ZipParser(settings),
            MetadataOnlyParser(),
            LegacyOfficeParser(),
        ]
        self.parsers = parsers

    def parser_for(self, file_path: Path) -> BaseParser:
        for parser in self.parsers:
            if parser.supports(file_path):
                return parser
        raise UnsupportedFormatError("不支持的文件格式")
