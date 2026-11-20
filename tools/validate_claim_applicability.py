"""Compare per-claim evidence instructions on frozen native answer contexts."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import random

from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.generation.base import GenerationRequest
from tools.verify_scene_guard_cross_source import (
    AnswerService, CancelToken, FIXTURES, GoldAnswerJudge, INDEX_DB, OpenAICompatibleGateway,
    QueryInterpreter, ROOT, RagIndexStore, SETTINGS_PATH, WindowsCredentialSecretStore,
    load_gold_dataset, model_settings, read, sha,
)

PAST = ROOT / "reports/rag_scene_cross_source_20261108"
OUT = ROOT / "reports/rag_claim_applicability_20261108"
RULE = (
    "回答前按用户所问的各项内容逐一核对现有证据。"
    "对每项，先确定会改变答案的对象、条件或分支，再核对将要引用的来源是否直接支持该项事实。"
    "相似术语、编号或现象只提供线索。不同对象、检测方法、阶段或版本的操作和数值，"
    "只有资料明确支持其适用性时才能用于当前问题。通用原理或共用检查须有原文依据，并按资料的适用范围表达。"
    "条件不同的做法分别写清。步骤题保留原文明确的前置动作及顺序。"
    "有证据的子问题直接回答，逐项引用；其余明确说明现有资料能核实到哪里。"
    "只有用户要选定具体处理且缺少决定条件时，才询问该条件。全部所问内容均缺少适用证据时按证据不足处理。"
    "给出用户所问的回答及引文即可，无须展示内部核对过程。"
)
REPEATS = {
    "rag-biz-008": 1, "rag-biz-015": 3, "rag-biz-016": 1, "rag-biz-028": 3,
    "rag-biz-033": 3, "rag-biz-035": 1, "rag-biz-040": 1, "rag-biz-050": 1,
    "rag-row-holdout2-007": 1, "rag-biz-011": 1, "rag-biz-014": 1,
    "rag-biz-017": 1, "rag-biz-019": 1, "rag-biz-021": 1, "cross-scene-06": 1,
}


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def answer(job, case, source, settings, key):
    row = dict(job)
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    token = CancelToken()
    context = source["context_A"]
    entries = source["entries_A"]
    plan = AnswerService._plan_summary(QueryInterpreter().interpret(case.question))
    if job["arm"] == "B":
        plan += "\n" + RULE
    try:
        if context:
            request = GenerationRequest(
                question=case.question, context=context, query_plan_summary=plan,
                model=settings.rag_api_model, api_base_url=settings.rag_api_base_url,
                api_style=settings.rag_api_style, api_key=key,
                timeout_seconds=settings.rag_api_timeout_seconds,
            )
            output = gateway.generate(request, token).text.strip()
        else:
            output = "资料中没有足够依据。"
        row["text"] = output
        row["status"] = "insufficient" if output == "资料中没有足够依据。" else "answered"
        if row["status"] == "answered":
            citation = CitationValidator().inspect(output, {e["source_id"]: () for e in entries})
            row["citations"] = list(citation.cited_ids)
            if not citation.structurally_valid:
                row["citation_error"] = citation.reason
        row["judgement"] = GoldAnswerJudge(gateway, settings, key).evaluate(
            case, {"status": row["status"], "text": output, "evidence_context": context}, token)
    except Exception as exc:
        row["error"] = str(exc).replace(key, "[REDACTED]")[:500]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    settings = model_settings()
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    assert key
    store = RagIndexStore(INDEX_DB.parent / "rag")
    bundle = store.active()
    earlier = read(PAST / "protocol.json")
    assert earlier["bundle"] == bundle.bundle_id and earlier["model"] == settings.rag_api_model
    assert earlier["hashes"][str(SETTINGS_PATH)] == sha(SETTINGS_PATH)
    cases = {c.case_id: c for path in [*FIXTURES, PAST / "holdout.json"]
             for c in load_gold_dataset(path, reviewed_only=False).cases}
    prior = PAST / "cases.jsonl"
    native = {r["id"]: r for r in lines(prior)}
    assert set(REPEATS) <= native.keys() & cases.keys()
    assert all(cases[i].question == native[i]["question"] for i in REPEATS)
    hashes = {str(p): sha(p) for p in [
        Path(__file__).resolve(), prior, PAST / "protocol.json", PAST / "holdout.json", *FIXTURES,
        SETTINGS_PATH, store.manifest_path, bundle.metadata_path,
        *(ROOT / "local_full_text_search/rag").rglob("*.py"),
    ]}
    assert all(sha(p) == digest for p, digest in earlier["hashes"].items())
    jobs = [{"id": cid, "arm": arm, "repeat": repeat, "cohort": native[cid]["cohort"]}
            for cid, n in REPEATS.items() for repeat in range(1, n + 1)
            for arm in (("A", "B") if native[cid]["context_A"] else ("AB",))]
    random.Random(87124).shuffle(jobs)
    OUT.mkdir(exist_ok=True)
    protocol = {
        "bundle": bundle.bundle_id, "model": settings.rag_api_model, "hashes": hashes, "rule_B": RULE,
        "one_change": "Prompt at the answer layer only. Both arms use the saved native context A, "
                      "same query summary, generator, citation inspection and independent Gold judge.",
        "boundary": "Development isolation through the generator, not end-to-end acceptance. "
                    "Case ids are used only to schedule and judge experiments.",
        "gate": "Escalate only if multiple distinct business groups show verified, source-bound gains, "
                "at least one stable gain on a triplicate case, and zero verified new substantive errors, "
                "including spurious refusal or clarification, across the diagnostic cases. "
                "One-shot gains are preliminary. Any confirmed new substantive failure stops this candidate.",
        "stop": "Run the fixed schedule once. No quality retries. Transport retries use the same policy. "
                "The same failed variant will not be tuned on this cohort and called independently validated.",
        "schedule": jobs,
    }
    with (OUT / "protocol.json").open("x", encoding="utf8") as file:
        json.dump(protocol, file, ensure_ascii=False, indent=2)
    write(OUT / "inputs.json", [{"id": cid, "question": native[cid]["question"],
                                "context": native[cid]["context_A"], "entries": native[cid]["entries_A"]}
                               for cid in REPEATS])
    with (OUT / "answers.jsonl").open("x", encoding="utf8") as file, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(answer, job, cases[job["id"]], native[job["id"]], settings, key) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            file.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            file.flush()
            print(row["id"], row["arm"], row["repeat"], row.get("judgement", {}).get("verdict"),
                  row.get("error", ""), flush=True)
    assert all(sha(p) == digest for p, digest in hashes.items())
    write(OUT / "completion.json", {"rows": len(jobs), "frozen_inputs_verified": True})


if __name__ == "__main__":
    main()
