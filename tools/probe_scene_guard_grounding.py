"""Recheck selector parity and isolate the answer effect of source prioritization."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from tools.answer_scene_guard import model_settings, run_answer, WindowsCredentialSecretStore
from tools.validate_scene_guard import (
    FIXTURES, OUT as FIRST, REPORTS, RagIndexStore, question_signature, select,
    read, sha, load_gold_dataset,
)

OUT = REPORTS / "rag_scene_guard_v2_20261108"


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def main():
    hashes = read(OUT / "protocol.json")["hashes"]
    assert all(sha(path) == digest for path, digest in hashes.items())
    original = {r["id"]: r for r in read(FIRST / "selection.json")}
    formal = {r["id"]: r for r in read(OUT / "selection.json")}
    assert all((row["context_A"], row["context_B"], row["entries_B"]) ==
               (original[cid]["context_A"], original[cid]["context_B"], original[cid]["entries_B"])
               for cid, row in formal.items())
    settings = model_settings()
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    queries = {h["id"]: h["question"] for h in read(FIRST / "holdout.json")}
    holdout = {r["id"]: r for r in read(FIRST / "holdout_selection.json")}
    with store.connect_read_only(bundle.metadata_path) as conn:
        for cid, row in holdout.items():
            rendered, entries, _ = select(
                queries[cid], row["context_A"], row["entries_A"], row["audit"]["ranked_candidates"],
                lambda key: store.load_chunk(conn, key), settings,
            )
            assert (rendered, entries) == (row["context_B"], row["entries_B"]), cid
    checks = [
        ("同批多个项目定标报DUP", ("DUP", "multiple")),
        ("只有一个项目校准报DEV", ("DEV", "single")),
        ("同批多个项目定标没有报DEV，怎么查", None),
        ("同批多个项目定标未报DEV，但有异常", None),
        ("多个项目校准无DEV报警", None),
        ("多个项目定标报DEV和DUP，如何区分", None),
        ("单个项目定标报DEV或DUP分别怎么办", None),
        ("单个项目和多个项目报DUP时有什么区别，均为定标", None),
        ("多个项目报DEV，但不是在定标阶段", None),
        ("多个项目定标有异常", None),
        ("定标报DEV，具体有多少个项目还不知道", None),
    ]
    assert all(question_signature(q) == expected for q, expected in checks)
    dump("selector_recheck.json", {
        "formal_and_regression_identical_to_v1": len(formal),
        "holdout_identical_to_v1": len(holdout),
        "boundary_checks": [{"question": q, "expected": s, "actual": question_signature(q)} for q, s in checks],
    })

    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or ""
    assert key
    cases = {c.case_id: c for p in (FIXTURES[0], FIRST / "holdout_gold.json")
             for c in load_gold_dataset(p, reviewed_only=False).cases}
    jobs = []
    for cid, row in [("rag-biz-019", formal["rag-biz-019"]),
                     ("scene-h01", holdout["scene-h01"]), ("scene-h04", holdout["scene-h04"])]:
        anchors = set(row["audit"]["selected"]["pair"])
        ids = tuple(e["source_id"] for e in row["entries_B"] if e["chunk_id"] in anchors)
        assert len(ids) == len(anchors)
        for rep in range(1, 4):
            jobs.append(((cases[cid], "C", rep, row["context_B"], row["entries_B"]), ids))
        if cid != "rag-biz-019":
            for arm in ("A", "B"):
                for rep in (2, 3):
                    jobs.append(((cases[cid], arm, rep, row[f"context_{arm}"], row[f"entries_{arm}"]), ()))
    dump("grounding_protocol.json", {
        "runner": sha(__file__), "answer_runner": sha(Path(__file__).with_name("answer_scene_guard.py")),
        "selector": sha(Path(__file__).with_name("validate_scene_guard.py")),
        "calls": len(jobs), "C": "Only a generic source prioritization instruction differs from B.",
        "roles": "019 is the reviewed target. The new DUP questions are diagnostic development after "
                 "inspecting their first generation outputs. They represent one business scenario.",
        "gate": "019 three correct answers, at least two of three for each DUP diagnostic, "
                "no cross-method or unsupported operational claim confirmed on manual review.",
    })
    with (OUT / "grounding_answers.jsonl").open("x", encoding="utf8") as file:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_answer, job, settings, key, scene_sources=ids) for job, ids in jobs]
            for future in as_completed(futures):
                row = future.result()
                file.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
                file.flush()
                print(row["id"], row["arm"], row["repeat"],
                      row.get("judgement", {}).get("verdict"), row.get("error"), flush=True)
    assert all(sha(path) == digest for path, digest in hashes.items())


if __name__ == "__main__":
    main()
