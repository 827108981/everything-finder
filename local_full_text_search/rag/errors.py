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


class RagSourceChangedError(RagError):
    pass


class RagCorpusTooLargeError(RagError):
    pass


class RagConfigurationError(RagError):
    pass


class RagGenerationError(RagError):
    pass
