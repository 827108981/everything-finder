"""Isolated comparison of scope-aware scene admission on a frozen RAG baseline."""
from collections import Counter
import argparse
import re
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk
from tools.validate_scene_quota import FIXTURES, REPORTS, ROOT, SCENE, delta, split_context
from tools.rag_eval_common import lines, read, sha
import json

OUT = REPORTS / "rag_scene_guard_20261108"
SCOPE = {
    "single": r"(?:仅|只有)?(?:单个|单一|一个|个别|某个)(?:发光|检测|免疫|生化)?项目",
    "multiple": r"(?:大部分|大多数|多数|部分|多个|多|数个|好几个|几个|多项)(?:发光|检测|免疫|生化)?项目|好几项",
    "all": r"(?:几乎)?(?:所有|全部)(?:发光|检测|免疫|生化)?项目",
}
ALARM = re.compile(r"(?:报警(?:为|代码)?|报)\s*[:：]?\s*([A-Z]{3,5})(?![A-Za-z0-9/])|(?<![A-Za-z0-9])([A-Z]{3,5})\s*报警")
NEGATED_ALARM = re.compile(r"(?:未|没有|没|不曾|并非|不是|无)\s*(?:发生|出现)?\s*$")
ADDITIONAL_CODE = re.compile(r"\s*(?:以及|和|与|及|或|、|/|／)\s*[A-Z]{3,5}(?![A-Za-z0-9])")
CALIBRATION = re.compile(r"定标|校准")
NEGATED_CALIBRATION = re.compile(r"(?:不是|并非|不属于)\s*(?:在|于)?\s*(?:定标|校准)")
NEGATED = re.compile(r"不是|并非|并不是|并不|不属于|非$|而非")


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def scope_matches(text):
    hits = []
    for scope, pattern in SCOPE.items():
        for m in re.finditer(pattern, text):
            prefix = text[max(m.start() - 5, 0):m.start()]
            suffix = text[m.end():m.end() + 5]
            if NEGATED.search(prefix) or re.match(r"(?:都|均)?(?:正常|通过|都过了|都没问题)", suffix):
                continue
            hits.append((scope, m.group()))
    return hits


def alarm_matches(text):
    matches = list(ALARM.finditer(text))
    for match in matches:
        if NEGATED_ALARM.search(text[max(0, match.start() - 10):match.start()]):
            return set()
        if ADDITIONAL_CODE.match(text, match.end()):
            return set()
    return {m.group(1) or m.group(2) for m in matches}


def question_signature(question):
    codes = alarm_matches(question)
    scopes = scope_matches(question)
    if (len(codes) == 1 and len({s for s, _ in scopes}) == 1
            and CALIBRATION.search(question) and not NEGATED_CALIBRATION.search(question)):
        return next(iter(codes)), scopes[0][0]
    return None


def source_signatures(text):
    # A heading or statement must bind scope and alarm in the same sentence/row.
    signatures = {}
    for line in re.split(r"[\n。；;？?]", text):
        scopes = scope_matches(line)
        codes = alarm_matches(line)
        if (len(codes) == 1 and len({s for s, _ in scopes}) == 1
                and CALIBRATION.search(line) and not NEGATED_CALIBRATION.search(line)):
            signatures[(next(iter(codes)), scopes[0][0])] = line.strip()
    return signatures


def relation(question, text):
    signatures = source_signatures(text)
    if len(signatures) != 1:
        return "unknown", list(signatures.values())
    key, quote = next(iter(signatures.items()))
    if key == question:
        return "match", [quote]
    # Same alarm with disjoint single versus plural scope is decisive.
    if key[0] == question[0] and (key[1] == "single") != (question[1] == "single"):
        return "scope_conflict", [quote]
    return "different_scene", [quote]


def select(question, context, entries, pairs, load, settings, *, allow_companion=False):
    sig = question_signature(question)
    audit = {"question_signature": sig, "ranked_candidates": [], "original": [], "selected": None}
    if not sig or not entries:
        return context, entries, audit
    for pair in pairs:
        state, quote = relation(sig, "\n".join(load(cid).text for cid in pair["ids"]))
        audit["ranked_candidates"].append({**pair, "relation": state, "quote": quote})
    audit["ranked_candidates"].sort(key=lambda p: p["relation"] != "match")
    audit["original"] = [
        {"chunk_id": e["chunk_id"], "state": state, "quote": quote}
        for e in entries
        for state, quote in [relation(sig, e["included_text"])]
    ]
    for pair in audit["ranked_candidates"]:
        if pair["relation"] != "match" or pair["reason"] not in ("document_cap3", "physical_top8", "available"):
            continue
        if pair["ce"] < settings.rag_min_rerank_score:
            continue
        chunks = [load(cid) for cid in pair["ids"]]
        if len({c.document_key for c in chunks}) != 1:
            continue
        doc = chunks[0].document_key
        same_document = [e for e in entries if load(e["chunk_id"]).document_key == doc]
        conflicts = {a["chunk_id"] for a in audit["original"] if a["state"] == "scope_conflict"}
        # Scope correction only removes an explicitly conflicting segment from the same document.
        remove = {e["chunk_id"] for e in same_document if e["chunk_id"] in conflicts}
        retained = [e for e in entries if e["chunk_id"] not in remove]
        present = {e["chunk_id"] for e in retained}
        missing = [c for c in chunks if c.stable_chunk_id not in present]
        if not missing:
            continue
        count = len(same_document) - len(remove) + len(missing)
        exact_heading_present = any(
            entry["chunk_id"] == chunks[0].stable_chunk_id
            and relation(sig, entry["included_text"])[0] == "match"
            for entry in retained
        )
        needs_single_companion = (
            allow_companion and exact_heading_present and len(chunks) == 2 and len(missing) == 1
            and missing[0].stable_chunk_id == chunks[1].stable_chunk_id
        )
        extra = int(len(chunks) == 2 and (bool(remove) or needs_single_companion))
        if count > settings.rag_max_chunks_per_document + extra:
            continue
        if len(retained) + len(missing) > settings.rag_final_top_k:
            continue
        added = ContextBuilder().build([EvidenceChunk(c) for c in missing], max_chars=settings.rag_context_max_chars)
        if len(added.entries) != len(missing) or any(e["truncated"] for e in added.entries):
            continue
        native_blocks = [
            (entry, block) for entry, block in zip(entries, split_context(context, entries))
            if entry["chunk_id"] not in remove
        ]
        new_blocks = list(zip(added.entries, split_context(added.text, added.entries))) + native_blocks
        actual, blocks = [], []
        for i, (entry, block) in enumerate(new_blocks, 1):
            actual.append({**entry, "source_id": f"S{i}"})
            blocks.append(f"[S{i}]" + block[block.index("]") + 1:])
        rendered = "\n\n".join(blocks)
        if len(rendered) > settings.rag_context_max_chars:
            continue
        audit["selected"] = {
            "pair": pair["ids"], "removed": sorted(remove),
            "added": [c.stable_chunk_id for c in missing], "document_count": count,
        }
        return rendered, actual, audit
    return context, entries, audit


def main(*, allow_companion=False):
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    scene_rows = lines(SCENE / "cases.jsonl")
    assert all(sha(p) == h for p, h in read(SCENE / "protocol.json")["hashes"].items())
    assert bundle.bundle_id == read(SCENE / "protocol.json")["bundle_id"]
    paths = [SETTINGS_PATH, store.manifest_path, Path(__file__), *FIXTURES,
             REPORTS / "rag_scene_guard_20261108/holdout.json",
             SCENE / "cases.jsonl", SCENE / "scene_index.json", SCENE / "050_approved_alternatives.json",
             *(ROOT / "local_full_text_search/rag").rglob("*.py")]
    hashes = {str(p): sha(p) for p in paths}
    dump("protocol.json", {
        "hashes": hashes, "bundle": bundle.bundle_id,
        "cases": {"reviewed": 48, "previously_used_pending": 30, "new_source_based_pending": 5},
        "comparison": "Freeze stored baseline and saved candidate scores. Audit scope independently. "
                      "Supplement an exact same-alarm/scope/calibration pair; remove only same-document "
                      "segments with an explicit opposite single/plural scope for that same alarm. "
                      "Allow one extra same-document physical segment only for a two-page match after "
                      "removing a confirmed conflict. Retain the global segment and character budgets.",
        "allow_single_companion_of_exact_existing_heading": allow_companion,
        "evidence_gate": "019 complete action quote and no Gold loss or new full marked negative on 78 cases. "
                         "Verify approved 050 alternative separately. Inspect every changed context.",
        "answer_gate": "Paired answer gain with source binding and no identified critical regression. "
                       "Run fresh native retrieval on new questions and report them separately.",
    })
    cases = {c.case_id: c for path in FIXTURES for c in load_gold_dataset(path, reviewed_only=False).cases}
    rows = []
    with store.connect_read_only(bundle.metadata_path) as conn:
        index = load_chunk_rows(conn)
        cache = {}

        def load(cid):
            if cid not in cache:
                cache[cid] = store.load_chunk(conn, cid)
            return cache[cid]

        for scene in scene_rows:
            cid = scene["id"]
            context = scene["context"] or ""
            new, selected, audit = select(
                scene["question"], context, scene["entries"], scene["trace"], load, settings,
                allow_companion=allow_companion,
            )
            evidence = delta(scene["entries"], selected, map_case_evidence(cases[cid], index))
            rows.append({
                "id": cid, "cohort": scene["cohort"], "question": scene["question"], "changed": new != context,
                "context_A": context, "context_B": new, "entries_A": scene["entries"], "entries_B": selected,
                "audit": audit, **evidence,
            })
    changed = [r for r in rows if r["changed"]]
    gate = (
        any(r["id"] == "rag-biz-019" and r["gained"] for r in changed)
        and all(not r["lost"] and not r["new_negatives"] for r in rows)
    )
    assert len(rows) == 78 and all(sha(p) == h for p, h in hashes.items())
    dump("selection.json", rows)
    summary = {
        "cohorts": dict(Counter(r["cohort"] for r in rows)), "evidence_gate": gate,
        "recognized_queries": [r["id"] for r in rows if r["audit"]["question_signature"]],
        "changed": [{k: r[k] for k in ("id", "gained", "lost", "new_negatives")} for r in changed],
    }
    dump("evidence_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-companion", action="store_true")
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    main(allow_companion=args.allow_companion)
