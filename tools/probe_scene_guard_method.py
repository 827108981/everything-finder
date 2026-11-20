"""Isolate removal of explicitly cross-method supplements after a scene is confirmed."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re

from tools.answer_scene_guard import model_settings, run_answer, WindowsCredentialSecretStore
from tools.probe_scene_guard_grounding import OUT
from tools.validate_scene_guard import FIXTURES, OUT as FIRST, load_gold_dataset, read, sha, split_context

METHODS = {
    "luminescence": re.compile(r"化学发光|发光(?:仪|项目|校准|定标|检测)"),
    "biochemistry": re.compile(r"生化(?:仪|项目|校准|定标|分析|检测)"),
}


def method_names(value):
    return {name for name, pattern in METHODS.items() if pattern.search(value)}


def apply_method_guard(question, context, entries, anchors):
    methods = method_names(question)
    if len(methods) != 1:
        return context, entries, []
    retained = []
    removed = []
    for entry, block in zip(entries, split_context(context, entries)):
        titles = [s["filename"] for s in entry.get("sources", [])] or [entry["filename"]]
        title_methods = [method_names(title) for title in titles]
        incompatible = all(len(found) == 1 and found.isdisjoint(methods) for found in title_methods)
        if incompatible and entry["chunk_id"] not in anchors:
            removed.append({"chunk_id": entry["chunk_id"], "filename": entry["filename"],
                            "location": entry["location"], "method": sorted(title_methods[0])})
        else:
            retained.append((entry, block))
    rebuilt, blocks = [], []
    for index, (entry, block) in enumerate(retained, 1):
        rebuilt.append({**entry, "source_id": f"S{index}"})
        blocks.append(f"[S{index}]" + block[block.index("]") + 1:])
    return "\n\n".join(blocks), rebuilt, removed


def main():
    hashes = read(OUT / "protocol.json")["hashes"]
    assert all(sha(p) == h for p, h in hashes.items())
    contexts = read(OUT / "selection.json") + read(FIRST / "holdout_selection.json")
    cases = {c.case_id: c for path in (*FIXTURES, FIRST / "holdout_gold.json")
             for c in load_gold_dataset(path, reviewed_only=False).cases}
    settings = model_settings()
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or ""
    assert key
    jobs, changes = [], []
    for row in contexts:
        if not row["changed"]:
            continue
        case = cases[row["id"]]
        pair = set(row["audit"]["selected"]["pair"])
        context, entries, removed = apply_method_guard(case.question, row["context_B"], row["entries_B"], pair)
        refs = tuple(e["source_id"] for e in entries if e["chunk_id"] in pair)
        assert len(refs) == len(pair)
        if context == row["context_B"]:
            continue
        changes.append({"id": case.case_id, "removed": removed, "entries": entries, "context": context})
        for repeat in range(1, 4):
            jobs.append(((case, "D", repeat, context, entries), refs))
    with (OUT / "method_guard_protocol.json").open("x", encoding="utf8") as file:
        json.dump({
            "comparison": "Same generation instruction as C. Remove only explicitly other-method title "
                          "sources outside the verified scene after exact scene admission.",
            "changes": changes, "jobs": len(jobs),
            "criterion": "No cross-method claim and all core facts on all three repeats.",
            "runner_sha256": sha(__file__),
        }, file, ensure_ascii=False, indent=2)
    with (OUT / "method_guard_answers.jsonl").open("x", encoding="utf8") as file, ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_answer, job, settings, key, scene_sources=refs) for job, refs in jobs]
        for future in as_completed(futures):
            row = future.result()
            file.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            file.flush()
            print(row["id"], row["arm"], row["repeat"],
                  row.get("judgement", {}).get("verdict"), row.get("error"), flush=True)
    assert all(sha(p) == h for p, h in hashes.items())


if __name__ == "__main__":
    main()
