from __future__ import annotations

from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.errors import UnsupportedFormatError
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.docx_parser import DocxParser
from local_full_text_search.parsers.image_parser import ImageParser
from local_full_text_search.parsers.legacy_office_parser import LegacyOfficeParser
from local_full_text_search.parsers.metadata_parser import MetadataOnlyParser
from local_full_text_search.parsers.ooxml.docx_stream_parser import DocxStreamParser
from local_full_text_search.parsers.ooxml.pptx_stream_parser import PptxStreamParser
from local_full_text_search.parsers.ooxml.xlsx_stream_parser import XlsxStreamParser
from local_full_text_search.parsers.pdf_parser import PdfParser
from local_full_text_search.parsers.pptx_parser import PptxParser
from local_full_text_search.parsers.text_parser import TextParser
from local_full_text_search.parsers.xlsx_parser import XlsxParser
from local_full_text_search.parsers.zip_parser import ZipParser
from local_full_text_search.ocr.ocr_engine import OcrEngine


class ParserRegistry:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        docx_parser: BaseParser = DocxParser()
        pptx_parser: BaseParser = PptxParser()
        xlsx_parser: BaseParser = XlsxParser()
        if settings.fast_ooxml_enabled:
            docx_parser = DocxStreamParser(fallback=docx_parser, defer_normalization=True)
            pptx_parser = PptxStreamParser(fallback=pptx_parser, defer_normalization=True)
            xlsx_parser = XlsxStreamParser(fallback=xlsx_parser, defer_normalization=True)
        shared_ocr = (
            OcrEngine(settings.ocr_language, settings.ocr_cpu_threads)
            if settings.enable_ocr and (settings.ocr_images or settings.ocr_scanned_pdf)
            else None
        )
        parsers: list[BaseParser] = [
            TextParser(),
            PdfParser(
                enable_scanned_ocr=settings.enable_ocr and settings.ocr_scanned_pdf,
                ocr_language=settings.ocr_language,
                parallel_min_bytes=settings.pdf_parallel_min_bytes,
                parallel_min_pages=settings.pdf_parallel_min_pages,
                parallel_workers=max(2, settings.parser_workers),
                ocr_engine=shared_ocr,
                ocr_cpu_threads=settings.ocr_cpu_threads,
            ),
            docx_parser,
            xlsx_parser,
            pptx_parser,
            ImageParser(
                language=settings.ocr_language,
                enabled=settings.enable_ocr and settings.ocr_images,
                min_pixels=settings.min_ocr_image_pixels,
                max_side=settings.max_ocr_image_side,
                ocr_engine=shared_ocr,
                ocr_cpu_threads=settings.ocr_cpu_threads,
            ),
            ZipParser(settings, ocr_engine=shared_ocr),
            MetadataOnlyParser(),
            LegacyOfficeParser(
                timeout_seconds=settings.single_file_timeout_seconds,
                conversion_cache=settings.legacy_conversion_cache,
                fast_ooxml=settings.fast_ooxml_enabled,
            ),
        ]
        self.parsers = parsers

    def parser_for(self, file_path: Path) -> BaseParser:
        for parser in self.parsers:
            if parser.supports(file_path):
                return parser
        raise UnsupportedFormatError("不支持的文件格式")
