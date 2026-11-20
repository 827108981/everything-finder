"""Frozen 2x2 prompt experiment. Gold only goes to the separate evaluator."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
import re
import time

from tools.run_simple_selection_api_20261118 import load_inputs, OUT as HISTORY, ROOT
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge, _identifier_binding_check, _verdict
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway, generation_system_prompt, provider_request_payload
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

OUT = ROOT / "reports/rag_prompt_factorial_20261119"
ARMS = ("baseline", "behavior", "completeness", "combined")
TARGETS = ("rag-biz-014", "rag-biz-016", "rag-biz-020", "rag-biz-021", "rag-biz-035",
           "rag-row-holdout-001", "rag-row-holdout2-001")


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def inputs():
    _, cases, contexts = load_inputs()
    settings = AppSettings.from_dict(json.loads(Path(SETTINGS_PATH).read_text(encoding="utf8")))
    return cases, contexts, settings


def legacy_guard(case, text, context):
    import unicodedata
    pattern = r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_.-]*\d)[A-Za-z][A-Za-z0-9]*(?:[_.-][A-Za-z0-9]+)*(?![A-Za-z0-9])"
    normalize = lambda s: unicodedata.normalize("NFKC", s).casefold()
    device_text = " ".join(c.source_text for c in QueryInterpreter().interpret(case.question).constraints if c.kind.value == "device")
    expected = sorted({normalize(x) for x in re.findall(pattern, case.question)} - {normalize(x) for x in re.findall(pattern, device_text)})
    sections = {m[1]: m[2] for m in re.finditer(r"(?m)^\[(S\d+)\]\s*\n([\s\S]*?)(?=^\[S\d+\]\s*$|\Z)", context)}
    actual = "\n".join(sections[s] for s in set(re.findall(r"\[(S\d+)\]", text)) if s in sections)
    absent = [x for x in expected if not re.search(r"(?<![A-Za-z0-9])" + re.escape(x) + r"(?![A-Za-z0-9])", normalize(actual))]
    return {"status": "review" if absent else "pass", "identifiers_missing_from_cited_sources": absent}


def recheck(case, judgement, text, context, *, legacy=False):
    # Reuse the same semantic verdicts; isolate only deterministic guard changes.
    result = dict(judgement)
    rubrics = [r for r in result["rubrics"] if r["id"] != "eval_identifier_reference"]
    guard = None
    if case.expected_behavior in {"answer", "partial_answer"}:
        guard = (legacy_guard if legacy else _identifier_binding_check)(case, text, context)
        if guard["status"] == "review":
            rubrics.append(dict(id="eval_identifier_reference", kind="source_binding", role="core",
                gold_text="明确编号需在实际引用中核实", status="uncertain",
                reason="明确编号未在实际引用中核实: " + ", ".join(guard["identifiers_missing_from_cited_sources"])))
    verdict, findings = _verdict(case, result["behavior"], rubrics)
    return {**result, "rubrics": rubrics, "identifier_check": guard, "verdict": verdict, "findings": findings}


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        check_frozen()
        return
    cases, contexts, settings = inputs()
    protected = [Path(SETTINGS_PATH), HISTORY / "real_entry_simple.jsonl", HISTORY / "simple_native.jsonl",
                 HISTORY / "api_answers.jsonl", HISTORY / "protocol.json",
                 ROOT / "tests/rag/fixtures/rag_active_suite_v5_1.json"]
    protected += list((ROOT / "tests/rag/fixtures").glob("*.jsonl")) + list((ROOT / "tests/rag/fixtures").glob("*_review.json"))
    protected += [ROOT / p for p in ("local_full_text_search/rag/answer_service.py", "local_full_text_search/rag/context_builder.py",
        "local_full_text_search/rag/evidence_policy.py", "local_full_text_search/rag/reranking/local_bge.py",
        "local_full_text_search/rag/generation/openai_compatible.py", "local_full_text_search/rag/generation/base.py",
        "local_full_text_search/rag/evaluation/answer_judge.py")]
    prompts = {arm: generation_system_prompt(arm) for arm in ARMS}
    protocol = dict(bundle_id="g000033-71ec3c52f3", cases=111, arms=list(ARMS), targets=list(TARGETS),
        first_round_cases=111, targeted_rounds=3, expected_generation_tasks=500, expected_judge_tasks=500,
        workers=4, shuffle_seed=20261119, model=settings.rag_api_model, api_style=settings.rag_api_style,
        adapter_enabled=settings.rag_query_plan_enabled, answer_review_enabled=settings.rag_answer_review_enabled,
        semantic_clarification_enabled=settings.rag_semantic_clarification_enabled,
        experiment="Frozen real Context generator-only 2x2; real service entry verified separately",
        changed_factors=["answer behavior system instructions", "completeness and citation system instructions"],
        frozen_context_sha256={k: sha256(v["context"].encode()).hexdigest() for k, v in contexts.items()},
        prompts=prompts, protected={str(p): sha(p) for p in protected},
        excluded_repair_target="rag-biz-019 retained in all displays and fixed denominators; no upstream repair",
        judge_guard="Same semantic judge, preserve legacy and corrected identifier guard separately; no scope inheritance without explicit evidence",
        acceptance="No confirmed new factual, scope, refusal or necessary-core regression in baseline-correct cases; targeted gains stable over 3 runs; pending/disputed separate; no prompt tuning after start",
        stop="One complete fixed experiment; retry only transport/invalid judge JSON, not answer quality. A regression disqualifies general adoption.")
    assert not settings.rag_query_plan_enabled and not settings.rag_answer_review_enabled
    write(OUT / "protocol.json", protocol)
    previous = [json.loads(l) for l in (HISTORY / "api_answers.jsonl").read_text(encoding="utf8").splitlines()]
    changes = []
    for r in previous:
        corrected = recheck(cases[r["id"]], r["judgement"], r["draft_answer"], contexts[r["id"]]["context"])
        changes.append(dict(id=r["id"], old=r["judgement"]["verdict"], corrected=corrected["verdict"],
                            identifier_check=corrected.get("identifier_check"), semantic_judge_reused=True))
    write(OUT / "evaluation_only_audit.json", dict(records=changes, api_calls=0,
        changed=[r for r in changes if r["old"] != r["corrected"]]))
    write(OUT / "step_status.json", {"freeze": "通过", "unit_tests": "通过（67项）", "generation": "待核实",
        "scope_050": "待核实：原PPT含多个型号，其他页出现目标型号不足以证明第17张适用，未自动继承全书型号"})
    print("PROTOCOL_FROZEN", {"cases": len(cases), "tasks": 500, "prompts": list(prompts)}, flush=True)


def check_frozen():
    p = json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    drift = [path for path, digest in p["protected"].items() if sha(path) != digest]
    assert not drift, drift
    return p


def run():
    initialize()
    protocol = check_frozen()
    cases, contexts, settings = inputs()
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    if not key:
        raise RuntimeError("No saved API credential")
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    judge = GoldAnswerJudge(gateway, settings, key)
    output = OUT / "answers.jsonl"
    prior = [json.loads(l) for l in output.read_text(encoding="utf8").splitlines()] if output.exists() else []
    done = {(r["id"], r["arm"], r["round"]) for r in prior}
    tasks = [(case_id, arm, 1) for case_id in cases for arm in ARMS]
    tasks += [(case_id, arm, repeat) for case_id in TARGETS for arm in ARMS for repeat in (2, 3)]
    random.Random(protocol["shuffle_seed"]).shuffle(tasks)
    assert len(tasks) == 500
    tasks = [task for task in tasks if task not in done]

    def worker(task):
        case_id, arm, repeat = task
        case, ctx = cases[case_id], contexts[case_id]
        request = GenerationRequest(question=case.question, context=ctx["context"], model=settings.rag_api_model,
            api_base_url=settings.rag_api_base_url, api_style=settings.rag_api_style, api_key=key,
            timeout_seconds=settings.rag_api_timeout_seconds, max_output_tokens=1000,
            query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(case.question)), prompt_variant=arm)
        payload = provider_request_payload(request)
        record = dict(id=case_id, arm=arm, round=repeat, question=case.question, review_status=case.review_status,
            expected_behavior=case.expected_behavior, context_sha256=sha256(ctx["context"].encode()).hexdigest(),
            context_chars=len(ctx["context"]), system_prompt_sha256=sha256(generation_system_prompt(arm).encode()).hexdigest(),
            model=request.model, api_request=payload, generator_only=True, gold_in_generation=False)
        assert record["context_sha256"] == protocol["frozen_context_sha256"][case_id]
        try:
            answer = gateway.generate(request, CancelToken())
            record["answer"] = answer.text
            citation = CitationValidator().inspect(answer.text, {e["source_id"]: () for e in ctx["entries"]})
            record["citation_check"] = dict(structurally_valid=citation.structurally_valid, cited_ids=list(citation.cited_ids),
                unknown_ids=list(citation.unknown_ids), reason=citation.reason)
            record["judgement"] = judge.evaluate(case, dict(status="answered", text=answer.text, evidence_context=ctx["context"]), CancelToken())
            record["legacy_guard_judgement"] = recheck(case, record["judgement"], answer.text, ctx["context"], legacy=True)
        except Exception as exc:
            record["error"] = str(exc).replace(key, "[REDACTED]")[:600]
        return record

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool, output.open("a", encoding="utf8") as stream:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            stream.write(json.dumps(record, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            stream.flush()
            prior.append(record)
            print("RESULT", len(prior), "/500", record["id"], record["arm"], record["round"],
                  record.get("judgement", {}).get("verdict", "error"), record.get("error", ""), flush=True)
    check_frozen()
    write(OUT / "execution.json", dict(tasks=len(prior), errors=[{k:r[k] for k in ("id","arm","round","error")} for r in prior if "error" in r],
        elapsed_seconds=time.monotonic()-start, frozen_after=True))
    print("EXPERIMENT_COMPLETE", len(prior), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("initialize", "run", "check"))
    action = parser.parse_args().action
    {"initialize": initialize, "run": run, "check": check_frozen}[action]()
