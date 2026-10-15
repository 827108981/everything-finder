from __future__ import annotations

from local_full_text_search.rag.models import Citation, RagAnswer


class EvidencePolicy:
    @staticmethod
    def insufficient(
        trace_id: str,
        generation: int | None,
        citations: tuple[Citation, ...] = (),
    ) -> RagAnswer:
        return RagAnswer(
            status="insufficient",
            answer_text="资料库中没有足够依据回答这个问题。",
            citations=citations,
            trace_id=trace_id,
            source_generation=generation,
        )
