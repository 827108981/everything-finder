"""Run post-freeze source-based acceptance controls, separate from development."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from tools.answer_scene_guard import fresh_holdout, model_settings, run_answer, WindowsCredentialSecretStore
from tools.validate_scene_guard import RagIndexStore, read, sha
from tools.validate_scene_quota import REPORTS, canonical_content
from tools.validate_structured_material import slides, text

OUT = REPORTS / "rag_scene_guard_v2_20261108/acceptance"


def main():
    cases_path = OUT / "holdout.json"
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    settings = model_settings()
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or ""
    assert key
    files = {}
    for case in read(cases_path):
        path = Path(case.get("file", "E:/生化免疫临床知识/化学发光定标处理思路分析.pptx"))
        page = next(text(s) for number, s in slides(path) if number == case["page"])
        assert canonical_content(case["quote"]) in canonical_content(page), case["id"]
        files[str(path)] = sha(path)
    frozen = {
        str(p): sha(p) for p in (
            cases_path, Path(__file__), Path(__file__).with_name("answer_scene_guard.py"),
            Path(__file__).with_name("validate_scene_guard.py"),
        )
    } | files
    protocol = {
        "hashes": frozen, "bundle": bundle.bundle_id,
        "comparisons": "A native retrieval and answer. C guarded scene selection and source prioritization. "
                       "Reuse identical requests once. Pending controls do not enter formal totals.",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf8")
    cases, contexts = fresh_holdout(settings, store, bundle, input_path=cases_path, output_path=OUT)
    assert all(not row["lost"] and not row["new_negatives"] for row in contexts)
    with (OUT / "answers.jsonl").open("x", encoding="utf8") as file, ThreadPoolExecutor(max_workers=4) as pool:
        tasks = []
        for row in contexts:
            case = cases[row["id"]]
            if not row["changed"]:
                tasks.append(pool.submit(run_answer, (case, "AC", 1, row["context_A"], row["entries_A"]), settings, key))
                continue
            anchors = set(row["audit"]["selected"]["pair"])
            refs = tuple(e["source_id"] for e in row["entries_B"] if e["chunk_id"] in anchors)
            for arm, key_arm in (("A", "A"), ("C", "B")):
                job = (case, arm, 1, row[f"context_{key_arm}"], row[f"entries_{key_arm}"])
                tasks.append(pool.submit(run_answer, job, settings, key, scene_sources=refs if arm == "C" else ()))
        for task in as_completed(tasks):
            row = task.result()
            file.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            file.flush()
            print(row["id"], row["arm"], row.get("judgement", {}).get("verdict"), row.get("error"), flush=True)
    assert all(sha(p) == h for p, h in frozen.items())


if __name__ == "__main__":
    main()
