"""Evaluate the one-companion exception and apply the frozen scene answer protocol."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from tools.answer_scene_guard import fresh_holdout, model_settings, run_answer, WindowsCredentialSecretStore
from tools.probe_scene_guard_method import apply_method_guard
from tools.validate_scene_guard import (
    FIXTURES, OUT as FIRST, REPORTS, RagIndexStore, delta, load_chunk_rows, load_gold_dataset,
    map_case_evidence, read, select, sha,
)
from tools.validate_scene_quota import canonical_content
from tools.validate_structured_material import slides, text

OUT = REPORTS / "rag_scene_guard_final_20261108"
PREVIOUS = REPORTS / "rag_scene_guard_v2_20261108"


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def main():
    settings = model_settings()
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    frozen = read(OUT / "protocol.json")["hashes"]
    assert all(sha(p) == digest for p, digest in frozen.items())
    inputs = [OUT / "holdout.json", Path(__file__), Path(__file__).with_name("answer_scene_guard.py"),
              Path(__file__).with_name("probe_scene_guard_method.py")]
    source = Path("E:/生化免疫临床知识/化学发光定标处理思路分析.pptx")
    pages = {n: text(s) for n, s in slides(source)}
    assert all(canonical_content(r["quote"]) in canonical_content(pages[r["page"]])
               for r in read(OUT / "holdout.json"))
    hashes = {str(p): sha(p) for p in [source, *inputs]}
    dump("final_protocol.json", {
        "hashes": hashes, "gate": "No necessary evidence loss or new full negative. "
                                 "Confirm strict target evidence and answer quality, audit new answers.",
        "comparison": "Allow one physical companion for an exact scene heading already selected. "
                      "No evictions in that case. Keep the previously tested method guard and scene guidance.",
        "holdout": "One additional DUP wording and one distinct LVC control. "
                   "DUP wording belongs to the same business group and is not an independent group gain.",
    })
    cases, fresh = fresh_holdout(
        settings, store, bundle, input_path=OUT / "holdout.json", output_path=OUT, allow_companion=True
    )
    historical = read(OUT / "selection.json")
    reuse_paths = [FIRST, PREVIOUS / "acceptance"]
    for path in reuse_paths:
        cases.update({c.case_id: c for c in load_gold_dataset(path / "holdout_gold.json", reviewed_only=False).cases})
    cases.update({c.case_id: c for p in FIXTURES for c in load_gold_dataset(p, reviewed_only=False).cases})
    jobs, results = [], []
    with store.connect_read_only(bundle.metadata_path) as conn:
        index = load_chunk_rows(conn)

        def load(cid):
            return store.load_chunk(conn, cid)

        batch = historical + fresh + [r for p in reuse_paths for r in read(p / "holdout_selection.json")]
        for row in batch:
            case = cases[row["id"]]
            context, entries, audit = row["context_B"], row["entries_B"], row["audit"]
            if "cohort" not in row and row not in fresh:
                context, entries, audit = select(
                    case.question, row["context_A"], row["entries_A"], row["audit"]["ranked_candidates"],
                    load, settings, allow_companion=True,
                )
            anchors = set((audit["selected"] or {}).get("pair", []))
            removed = []
            if anchors:
                context, entries, removed = apply_method_guard(case.question, context, entries, anchors)
            scores = delta(row["entries_A"], entries, map_case_evidence(case, index))
            assert not scores["lost"] and not scores["new_negatives"], (case.case_id, scores)
            results.append({
                "id": case.case_id, "question": case.question,
                "context_A": row["context_A"], "context_final": context,
                "entries_A": row["entries_A"], "entries_final": entries,
                "selection": audit["selected"], "other_method_removed": removed, **scores,
            })
            if row in fresh or context != row["context_B"] and not removed:
                refs = tuple(e["source_id"] for e in entries if e["chunk_id"] in anchors)
                if context == row["context_A"]:
                    jobs.append(((case, "AE", 1, context, entries), ()))
                else:
                    for repeat in range(1, 4):
                        jobs.append(((case, "E", repeat, context, entries), refs))
                    if row in fresh:
                        for repeat in range(1, 4):
                            jobs.append(((case, "A", repeat, row["context_A"], row["entries_A"]), ()))
    dump("final_selection.json", results)
    dump("answer_schedule.json", [{"id": j[0].case_id, "arm": j[1], "repeat": j[2], "anchors": a}
                                   for j, a in jobs])
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or ""
    assert key
    with (OUT / "answers.jsonl").open("x", encoding="utf8") as file, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_answer, job, settings, key, scene_sources=refs) for job, refs in jobs]
        for task in as_completed(futures):
            row = task.result()
            file.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            file.flush()
            print(row["id"], row["arm"], row["repeat"],
                  row.get("judgement", {}).get("verdict"), row.get("error"), flush=True)
    assert all(sha(p) == digest for p, digest in (frozen | hashes).items())


if __name__ == "__main__":
    main()
