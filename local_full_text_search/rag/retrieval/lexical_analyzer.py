from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import unicodedata
from pathlib import Path
from functools import lru_cache

import jieba

from .lexical_models import LexicalPhrase, LexicalQuery, LexicalToken


ANALYZER_VERSION = "jieba-exact-v2"
_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9._/-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/-]*\d)[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*(?![A-Za-z0-9])"
)
_INDEXABLE_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")
_HAN_RE = re.compile(r"[\u3400-\u9fff]+")


def normalize_lexical_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


class LexicalAnalyzer:
    """查询和正文共用的通用分析器，不推断型号、模块或专业词角色。"""

    def __init__(self, *, hmm: bool = True) -> None:
        self.hmm = bool(hmm)
        self._tokenizer = jieba.Tokenizer()
        self._tokenizer.initialize()
        self.fingerprint = self._fingerprint()

    def _fingerprint(self) -> str:
        dictionary = Path(jieba.__file__).resolve().with_name("dict.txt")
        dictionary_hash = hashlib.sha256(dictionary.read_bytes()).hexdigest() if dictionary.is_file() else "missing"
        payload = {
            "analyzer_version": ANALYZER_VERSION,
            "jieba_version": importlib.metadata.version("jieba"),
            "dictionary_sha256": dictionary_hash,
            "hmm": self.hmm,
            "normalization": "NFKC+casefold",
            "identifier_rule": "mixed-alnum-boundary-v1",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def analyze_text(self, text: str) -> tuple[LexicalToken, ...]:
        original = str(text or "")
        tokens: list[LexicalToken] = []
        cursor = 0
        for match in _IDENTIFIER_RE.finditer(original):
            tokens.extend(self._plain_tokens(original[cursor:match.start()], cursor))
            normalized = normalize_lexical_text(match.group())
            if normalized:
                tokens.append(LexicalToken(match.group(), normalized, match.start(), match.end(), "identifier"))
            cursor = match.end()
        tokens.extend(self._plain_tokens(original[cursor:], cursor))
        return tuple(tokens)

    def _plain_tokens(self, text: str, base: int) -> list[LexicalToken]:
        output: list[LexicalToken] = []
        for surface, start, end in self._tokenizer.tokenize(text, mode="default", HMM=self.hmm):
            if not surface or not _INDEXABLE_RE.search(surface):
                continue
            normalized = normalize_lexical_text(surface).strip()
            if normalized and _INDEXABLE_RE.search(normalized):
                output.append(LexicalToken(surface, normalized, base + start, base + end, "word"))
        return output

    def analyze_query(self, text: str) -> LexicalQuery:
        all_tokens = self.analyze_text(text)
        unique: list[LexicalToken] = []
        duplicates: list[str] = []
        seen: set[str] = set()
        for token in all_tokens:
            if token.normalized in seen:
                duplicates.append(token.normalized)
                continue
            seen.add(token.normalized)
            unique.append(token)

        phrases: list[LexicalPhrase] = []
        phrase_seen: set[str] = set()
        # 长中文词本身也进入补充定位：同一连续原文在不同上下文中可能被
        # Jieba 切成不同词项，不能因为查询侧恰好切成一个词就失去兜底。
        for token in unique:
            if (
                token.surface_kind == "word" and 4 <= len(token.normalized) <= 12
                and _HAN_RE.fullmatch(token.normalized)
            ):
                phrase_seen.add(token.normalized)
                phrases.append(LexicalPhrase(
                    token.normalized, token.start, token.end, (token.normalized,)
                ))
        for han_match in _HAN_RE.finditer(str(text or "")):
            inside = [
                token for token in all_tokens
                if token.start >= han_match.start() and token.end <= han_match.end()
                and token.surface_kind == "word"
            ]
            for width in (2, 3):
                for index in range(0, len(inside) - width + 1):
                    group = inside[index:index + width]
                    start, end = group[0].start, group[-1].end
                    normalized = normalize_lexical_text(str(text or "")[start:end])
                    if not (4 <= len(normalized) <= 12) or normalized in phrase_seen:
                        continue
                    if not _HAN_RE.fullmatch(normalized):
                        continue
                    phrase_seen.add(normalized)
                    phrases.append(LexicalPhrase(
                        normalized, start, end, tuple(item.normalized for item in group)
                    ))
        return LexicalQuery(str(text or ""), tuple(unique), tuple(phrases), tuple(duplicates), self.fingerprint)

    @staticmethod
    def locator_bigrams(text: str) -> tuple[tuple[str, int, int], ...]:
        normalized = normalize_lexical_text(text)
        output: list[tuple[str, int, int]] = []
        for match in _HAN_RE.finditer(normalized):
            run = match.group()
            for offset in range(max(0, len(run) - 1)):
                output.append((run[offset:offset + 2], match.start() + offset, match.start() + offset + 2))
        return tuple(output)


@lru_cache(maxsize=2)
def get_lexical_analyzer(hmm: bool = True) -> LexicalAnalyzer:
    """复用只读分词器，避免每道题重复加载约 20 MB 的默认词典。"""

    return LexicalAnalyzer(hmm=hmm)
