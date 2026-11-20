"""Replay frozen scene candidates with two allocations of the same document cap."""
from collections import Counter
import json
from pathlib import Path
import re
from types import SimpleNamespace

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk
from tools.rag_eval_common import ROOT, lines, read, sha

REPORTS = ROOT / "reports"
OLD = REPORTS / "rag_evidence_selection_20261104"
SCENE = REPORTS / "rag_scene_entry_20261105"
OUT = REPORTS / "rag_scene_quota_20261108"
FIXTURES = [
    ROOT / "tests/rag/fixtures" / name for name in (
        "rag_gold_v4_2_review.json", "rag_acceptance_v1_review.json",
        "rag_row_holdout_v1_review.json", "rag_row_holdout_v2_review.json",
    )
]
RECORDS = REPORTS / "rag_retrieval_ab_results_20261103/resolved_cases.jsonl"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def dump_rows(name, rows):
    (OUT / name).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8"
    )


def score(entries, mappings):
    # Source aliases only count when the actual transmitted text contains the quote.
    stage = [
        {
            **entry,
            "member_chunk_ids": sorted(
                set(entry.get("member_chunk_ids", []))
                | {source["chunk_id"] for source in entry.get("sources", [])}
            ),
        }
        for entry in entries
    ]
    return score_stage("context", stage, mappings)


def delta(before_entries, after_entries, mappings):
    before, after = (score(entries, mappings) for entries in (before_entries, after_entries))
    lost = sorted(set(before["gold_completion_ranks"]) - set(after["gold_completion_ranks"]))
    gained = sorted(set(after["gold_completion_ranks"]) - set(before["gold_completion_ranks"]))
    negatives = sorted(set(after["hard_negative_ids"]) - set(before["hard_negative_ids"]))
    quotes = {m.evidence_id: m.quote for m in mappings}
    identical_text_elsewhere = {}
    for evidence_id in lost:
        same_text = [
            {k: e[k] for k in ("chunk_id", "filename", "location")}
            for e in after_entries
            if canonical_content(quotes[evidence_id]) in canonical_content(e["included_text"])
        ]
        if same_text:
            identical_text_elsewhere[evidence_id] = same_text
    return {
        "before_gold": sorted(before["gold_completion_ranks"]),
        "after_gold": sorted(after["gold_completion_ranks"]),
        "lost": lost, "gained": gained, "new_negatives": negatives,
        "identical_text_elsewhere_pending": identical_text_elsewhere,
        "page_only_negatives_after": after["hard_negative_page_only_ids"],
    }


def audit_old(load, maps):
    originals = {(r["id"], r["arm"]): r for r in lines(RECORDS)}
    grouped = {
        (r["id"], r["arm"]): r for r in lines(OLD / "changed_contexts.jsonl")
    }
    output = []
    for row in lines(OLD / "cases.jsonl"):
        key = (row["id"], row["arm"])
        saved = originals[key]
        before = saved["context_entries"]
        if row["method"] == "grouped_mmr":
            after = grouped[key]["context_entries"] if row["changed_context"] else before
        else:
            # Replay the old recorded final order. Only the renderer reads these objects.
            snaps = {s["chunk_id"]: s for s in saved["candidate_stages"]["constraints"]}
            items = []
            for chunk_id in row["after_ids"]:
                snap = snaps[chunk_id]
                chunk = load(chunk_id)
                assert chunk.text == snap["text"]
                item = EvidenceChunk(chunk, sources=tuple(load(s["chunk_id"]).source for s in snap["sources"]))
                if decision := snap.get("constraint_decision"):
                    item.constraint_decision = SimpleNamespace(
                        support_status=decision["support_status"],
                        judgements=tuple(SimpleNamespace(constraint=SimpleNamespace(
                            kind=SimpleNamespace(value=j["kind"])
                        )) for j in decision["judgements"]),
                    )
                items.append(item)
            rendered = ContextBuilder().build(items, max_chars=12_000)
            after = rendered.entries
            assert len(rendered.text) == row["chars"], key
        assert [e["chunk_id"] for e in after] == row["after_ids"], (key, row["method"])
        output.append({
            "id": row["id"], "upstream": row["arm"], "method": row["method"],
            "cohort": row["cohort"], "changed": row["changed_context"],
            **delta(before, after, maps[row["id"]]),
        })
    assert len(originals) == 236 and len(output) == 472
    return output


def split_context(text, entries):
    parts = re.split(r"(?m)^\[(S\d+)\]\n", text)
    assert parts[0] == ""
    assert parts[1::2] == [e["source_id"] for e in entries]
    blocks = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        if i + 2 < len(parts):
            body = body.removesuffix("\n\n")
        blocks.append(f"[{parts[i]}]\n{body}")
    assert "\n\n".join(blocks) == text
    return blocks


def select(base, trace, load, settings, *, reallocate):
    entries = base["entries"]
    context = base["context"] or ""
    # Existing scene trace is the immutable input. It already applies upstream hard
    # conflicts, score thresholds, behavior and total-slot checks in this order.
    log = []
    if entries:
        present = {e["chunk_id"] for e in entries}
        for hit in trace:
            reason = hit["reason"]
            if reason != "document_cap3":
                assert reason, "Saved control unexpectedly admitted a scene"
                continue
            missing = [load(i) for i in hit["ids"] if i not in present]
            assert missing and len(entries) + len(missing) <= settings.rag_final_top_k
            document = missing[0].document_key
            assert all(c.document_key == document for c in missing)
            same_doc = [i for i, e in enumerate(entries) if load(e["chunk_id"]).document_key == document]
            excess = len(same_doc) + len(missing) - settings.rag_max_chunks_per_document
            assert excess > 0
            if not reallocate or excess > len(same_doc):
                log.append({"pair": hit["ids"], "reason": reason})
                continue
            # Keep surviving native blocks byte-identical except for source labels.
            # Keep any already-present member of the selected scene together with its continuation.
            evictable = [i for i in same_doc if entries[i]["chunk_id"] not in hit["ids"]]
            if len(evictable) < excess:
                log.append({"pair": hit["ids"], "reason": "pair_members_use_all_slots"})
                continue
            remove = set(evictable[-excess:])
            added = ContextBuilder().build(
                [EvidenceChunk(chunk) for chunk in missing],
                max_chars=settings.rag_context_max_chars,
            )
            if len(added.entries) != len(missing) or any(e["truncated"] for e in added.entries):
                log.append({"pair": hit["ids"], "reason": "partial_scene"})
                continue
            kept = [(e, b) for i, (e, b) in enumerate(zip(entries, split_context(context, entries)))
                    if i not in remove]
            kept.extend(zip(added.entries, split_context(added.text, added.entries)))
            new_entries, blocks = [], []
            for index, (entry, block) in enumerate(kept, 1):
                source_id = f"S{index}"
                new_entries.append({**entry, "source_id": source_id})
                blocks.append(f"[{source_id}]" + block[block.index("]") + 1:])
            new_context = "\n\n".join(blocks)
            if len(new_context) > settings.rag_context_max_chars:
                log.append({"pair": hit["ids"], "reason": "char_budget"})
                continue
            log.append({
                "pair": hit["ids"], "reason": "reallocated",
                "added": [c.stable_chunk_id for c in missing],
                "removed": [entries[i]["chunk_id"] for i in sorted(remove)],
            })
            return new_context, new_entries, log
    return context, entries, log


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "selection.jsonl").exists(), "Use a distinct output directory for reruns"
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    assert (settings.rag_final_top_k, settings.rag_max_chunks_per_document,
            settings.rag_context_max_chars) == (8, 3, 12000)
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    previous = read(SCENE / "protocol.json")
    assert bundle.bundle_id == previous["bundle_id"]
    native_path = REPORTS / "rag_row_ab_confirm_20261105/resolved_cases.json"
    paths = [
        SETTINGS_PATH, store.manifest_path, Path(__file__), *FIXTURES, RECORDS,
        OLD / "cases.jsonl", OLD / "changed_contexts.jsonl", native_path,
        SCENE / "cases.jsonl", SCENE / "protocol.json",
        *(ROOT / "local_full_text_search/rag").rglob("*.py"),
    ]
    hashes = {str(p): sha(p) for p in paths}
    dump("protocol.json", {
        "bundle": bundle.bundle_id, "hashes": hashes, "source_scorer": "source-quote-v3",
        "A": "Preserve occupied per-document slots; reuse scene decisions from November 5.",
        "B": "At a document-cap rejection, reserve slots for the first feasible adjacent scene "
             "by replacing the minimum number of the lowest native entries from that document. "
             "Keep any existing scene member. Append original physical pages in original order.",
        "fixed": "Saved native context, scene candidate IDs, order, scores and prior conditions, "
                 "one pair, max 3 same-document segments, max 8 segments, max 12000 chars.",
        "gate": "Require 019 complete action quote, no new complete marked negative and no "
                "unresolved loss of an applicable critical fact before calling generation.",
        "scope": "Replay 78 previously used questions (48 formal, 30 pending). Re-score 472 "
                 "historic decisions separately. No Gold inputs to selection.",
    })
    cases = {c.case_id: c for path in FIXTURES for c in load_gold_dataset(path, reviewed_only=False).cases}
    with store.connect_read_only(bundle.metadata_path) as conn:
        chunks = load_chunk_rows(conn)
        cache = {}

        def load(chunk_id):
            if chunk_id not in cache:
                cache[chunk_id] = store.load_chunk(conn, chunk_id)
            assert cache[chunk_id] is not None, chunk_id
            return cache[chunk_id]

        maps = {cid: map_case_evidence(c, chunks) for cid, c in cases.items()
                if c.review_status != "disputed"}
        historical = audit_old(load, maps)
        dump_rows("historical_rescoring.jsonl", historical)
        print(f"Replayed {len(historical)} old decisions", flush=True)
        native = {r["id"]: r for r in read(native_path) if r["arm"] == "baseline"}
        results = []
        for scene in lines(SCENE / "cases.jsonl"):
            cid = scene["id"]
            base = native[cid]
            assert scene["question"] == base["question"] and scene["entries"] == base["entries"]
            left, before, control_trace = select(base, scene["trace"], load, settings, reallocate=False)
            right, after, trial_trace = select(base, scene["trace"], load, settings, reallocate=True)
            assert left == scene["context"]
            assert len(after) <= settings.rag_final_top_k and len(right) <= settings.rag_context_max_chars
            docs = Counter(load(e["chunk_id"]).document_key for e in after)
            assert max(docs.values(), default=0) <= settings.rag_max_chunks_per_document
            results.append({
                "id": cid, "cohort": scene["cohort"], "question": base["question"],
                "expected_behavior": cases[cid].expected_behavior, "changed": left != right,
                "context_A": left, "context_B": right, "entries_A": before, "entries_B": after,
                "trace_A": control_trace, "trace_B": trial_trace,
                **delta(before, after, maps[cid]),
            })
    assert len(results) == len(native) == 78
    assert all(sha(p) == h for p, h in hashes.items())
    dump_rows("selection.jsonl", results)
    changed = [r for r in results if r["changed"]]
    summary = {
        "historical_decisions": len(historical), "scene_pairs": len(results),
        "changed": [r["id"] for r in changed],
        "with_gains": [r["id"] for r in changed if r["gained"]],
        "with_losses": [r["id"] for r in changed if r["lost"]],
        "with_new_negatives": [r["id"] for r in changed if r["new_negatives"]],
        "all_caps_passed": True, "control_matches_prior_context": True,
        "inputs_unchanged": True,
    }
    dump("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
