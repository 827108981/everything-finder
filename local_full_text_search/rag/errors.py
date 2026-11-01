class RagError(Exception):
    """Base error shown by RAG workers as an actionable message."""


class RagModelMissingError(RagError):
    pass


class RagModelValidationError(RagError):
    pass


class RagIndexMissingError(RagError):
    pass


class RagIndexStaleError(RagError):
    pass


class RagIndexCorruptError(RagError):
    pass


class LexicalIndexNotReadyError(RagError):
    """关键词 V2 索引缺失、损坏或与当前 AI bundle 不一致。"""

    pass


class VectorIndexNotReadyError(RagError):
    """向量单元 V2 索引缺失、损坏或与当前 AI bundle 不一致。"""

    pass


class RagSourceChangedError(RagError):
    pass


class RagCorpusTooLargeError(RagError):
    pass


class RagConfigurationError(RagError):
    pass


class RagGenerationError(RagError):
    pass
