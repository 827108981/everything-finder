from __future__ import annotations

import re

from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.retrieval.common import allowed_source_exists, load_evidence

_ASCII = re.compile(r"[a-z0-9][a-z0-9_.:/+-]{1,}", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_QUOTED = re.compile(r'[\"“]([^\"”]+)[\"”]')


class LexicalRetriever:
    def __init__(self, candidate_limit: int = 200) -> None:
        self.candidate_limit = max(20, min(1000, int(candidate_limit)))

    def retrieve(self, connection, query: RagQuery, top_n: int = 30):
        terms = extract_lexical_terms(query.question)
        if not terms:
            return []
        allowed_sql, allowed_params = allowed_source_exists(query)
        meta = connection.execute("SELECT fts_mode FROM rag_meta WHERE id=1").fetchone()
        fts_mode = str(meta[0]) if meta is not None else "like"
        if fts_mode == "trigram" and any(len(term) >= 3 for term in terms):
            rows = self._fts_candidates(connection, terms, allowed_sql, allowed_params)
        else:
            rows = self._like_candidates(connection, terms, allowed_sql, allowed_params)
        ranked: list[tuple[int, float, int]] = []
        for row in rows:
            normalized = str(row["normalized_text"] or "")
            matched = sum(1 for term in terms if term in normalized)
            if matched <= 0:
                continue
            bm25 = float(row["bm25_score"] or 0.0)
            ranked.append((matched, bm25, int(row["chunk_rowid"])))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        output = []
        seen: set[str] = set()
        for _matched, _bm25, rowid in ranked:
            evidence = load_evidence(connection, rowid, query)
            if evidence is None or evidence.chunk.stable_chunk_id in seen:
                continue
            output.append(evidence)
            seen.add(evidence.chunk.stable_chunk_id)
            if len(output) >= max(1, int(top_n)):
                break
        return output

    def _fts_candidates(self, connection, terms, allowed_sql, allowed_params):
        expressions = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms if len(term) >= 3]
        if not expressions:
            return self._like_candidates(connection, terms, allowed_sql, allowed_params)
        match_expression = " OR ".join(expressions)
        try:
            return connection.execute(
                "SELECT c.chunk_rowid,c.normalized_text,bm25(rag_chunks_fts) AS bm25_score "
                "FROM rag_chunks_fts JOIN rag_chunks c ON c.chunk_rowid=rag_chunks_fts.rowid "
                f"WHERE rag_chunks_fts MATCH ? AND {allowed_sql} "
                "ORDER BY bm25_score ASC LIMIT ?",
                (match_expression, *allowed_params, self.candidate_limit),
            ).fetchall()
        except Exception:
            # MATCH syntax or tokenizer failures should leave an explicit slower
            # path rather than turn a recoverable lexical miss into an app crash.
            return self._like_candidates(connection, terms, allowed_sql, allowed_params)

    def _like_candidates(self, connection, terms, allowed_sql, allowed_params):
        useful = list(dict.fromkeys(term for term in terms if term))
        predicates = " OR ".join("c.normalized_text LIKE ? ESCAPE '!'" for _ in useful)
        patterns = [f"%{_escape_like(term)}%" for term in useful]
        return connection.execute(
            "SELECT c.chunk_rowid,c.normalized_text,0.0 AS bm25_score FROM rag_chunks c "
            f"WHERE ({predicates}) AND {allowed_sql} ORDER BY c.chunk_rowid LIMIT ?",
            (*patterns, *allowed_params, self.candidate_limit),
        ).fetchall()


def extract_lexical_terms(question: str, max_terms: int = 12) -> list[str]:
    normalized = normalize_text(question)
    priorities: list[str] = []
    for phrase in _QUOTED.findall(question):
        value = normalize_text(phrase)
        if len(value) >= 2:
            priorities.append(value)
    for match in _ASCII.findall(normalized):
        if len(match) >= 2:
            priorities.append(match)
    for run in _CJK_RUN.findall(normalized):
        if len(run) <= 6:
            priorities.append(run)
        else:
            grams = [run[index : index + 3] for index in range(len(run) - 2)]
            # Spread long Chinese sentences across the whole question while
            # retaining a deterministic order and a bounded MATCH expression.
            slots = min(8, len(grams))
            selected = [round(index * (len(grams) - 1) / max(1, slots - 1)) for index in range(slots)]
            priorities.extend(grams[index] for index in selected)
    seen: set[str] = set()
    result: list[str] = []
    for term in priorities:
        value = normalize_text(term)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
        if len(result) >= max_terms:
            break
    return result


def _escape_like(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")
