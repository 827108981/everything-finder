"""Local retrieval augmented generation services.

Heavy optional dependencies (Chroma, PyTorch and OpenAI) are imported only by
the concrete adapters so the existing keyword search remains usable without
installing the optional RAG environment.
"""

from .models import RagAnswer, RagQuery

__all__ = ["RagAnswer", "RagQuery"]
