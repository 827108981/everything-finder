"""Generate and separately judge answers for the frozen simple-selection contexts."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/rag_simple_selection_20261118"
FIXTURES = ROOT / "tests/rag/fixtures"
MANIFEST = FIXTURES / "rag_active_suite_v5_1.json"
HOLDOUT = FIXTURES / "rag_vector_source_holdout_20261116.jsonl"
OUTPUT = OUT / "api_answers.jsonl"


def load_inputs():
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    paths = [ROOT / d["path"] for d in manifest["datasets"]] + [HOLDOUT]
    cases = {}
    for path in paths:
        for case in load_gold_dataset(path, reviewed_only=False).cases:
            if case.case_id in cases:
                raise ValueError(f"Duplicate active case: {case.case_id}")
            cases[case.case_id] = case
    contexts = {r["id"]: r for r in (
        json.loads(line) for line in (OUT / "real_entry_simple.jsonl").read_text(encoding="utf8").splitlines()
    )}
    if set(cases) != set(contexts) or len(cases) != 111:
        raise ValueError(f"Active set/context mismatch: {len(cases)} cases, {len(contexts)} contexts")
    return manifest, cases, contexts


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite API results: {OUTPUT}")
    manifest, cases, contexts = load_inputs()
    settings = AppSettings.from_dict(json.loads(Path(SETTINGS_PATH).read_text(encoding="utf8")))
    if not settings.rag_api_model.strip():
        raise RuntimeError("No LLM model configured")
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    if not key:
        raise RuntimeError("No saved API credential")
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    judge = GoldAnswerJudge(gateway, settings, key)
    interpreter = QueryInterpreter()

    protocol = {
        "active_manifest": str(MANIFEST.relative_to(ROOT)),
        "active_manifest_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "holdout_sha256": sha256(HOLDOUT.read_bytes()).hexdigest(),
        "context_file_sha256": sha256((OUT / "real_entry_simple.jsonl").read_bytes()).hexdigest(),
        "bundle_id": manifest["bundle_id"],
        "case_count": len(cases),
        "context_selection": "Frozen actual AnswerService Context from simple arm; no new retrieval and no Gold injected into generation.",
        "generation_model": settings.rag_api_model,
        "api_host": settings.rag_api_base_url.split("/")[2] if "://" in settings.rag_api_base_url else "configured",
        "api_style": settings.rag_api_style,
        "answer_review_enabled": settings.rag_answer_review_enabled,
        "generation_and_judge_requests_expected": len(cases) * 2,
        "workers": 4,
        "gold_sent_to_generator": False,
        "gold_judge": "GoldAnswerJudge; separate API call after generation; verdict is an automatic judge, not human review.",
        "secrets_saved": False,
    }
    (OUT / "api_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    def run(case_id):
        case, row = cases[case_id], contexts[case_id]
        context = row["context"]
        sources = {e["source_id"]: () for e in row["entries"]}
        result = {
            "id": case_id,
            "review_status": case.review_status,
            "expected_behavior": case.expected_behavior,
            "question": case.question,
            "context_sha256": sha256(context.encode("utf8")).hexdigest(),
            "context_chars": len(context),
            "model": settings.rag_api_model,
            "answer_review_enabled": settings.rag_answer_review_enabled,
        }
        try:
            request = GenerationRequest(
                question=case.question,
                context=context,
                query_plan_summary=AnswerService._plan_summary(interpreter.interpret(case.question)),
                model=settings.rag_api_model,
                api_base_url=settings.rag_api_base_url,
                api_style=settings.rag_api_style,
                api_key=key,
                timeout_seconds=settings.rag_api_timeout_seconds,
            )
            generated = gateway.generate(request, CancelToken())
            result["draft_answer"] = generated.text
            citation = CitationValidator().inspect(generated.text, sources)
            answer = {
                "mode": "full",
                "text": generated.text,
                "status": "answered" if citation.structurally_valid else "citation_invalid",
                "source_ids": list(sources),
            }
            result["citation_check"] = {
                "structurally_valid": citation.structurally_valid,
                "cited_ids": list(citation.cited_ids),
                "unknown_ids": list(citation.unknown_ids),
                "reason": citation.reason,
            }
            result["judgement"] = judge.evaluate(
                case, {**answer, "evidence_context": context}, CancelToken()
            )
        except Exception as exc:
            result["error"] = str(exc).replace(key, "[REDACTED]")[:600]
        return result

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool, OUTPUT.open("x", encoding="utf8") as output:
        futures = {pool.submit(run, case_id): case_id for case_id in cases}
        for future in as_completed(futures):
            item = future.result()
            rows.append(item)
            serialized = json.dumps(item, ensure_ascii=False, default=str).replace(key, "[REDACTED]")
            output.write(serialized + "\n")
            output.flush()
            verdict = item.get("judgement", {}).get("verdict", "error" if item.get("error") else "unjudged")
            print(f"API_RESULT {len(rows)}/111 {item['id']} {item['review_status']} {verdict} {item.get('error', '')}", flush=True)

    rows.sort(key=lambda x: x["id"])
    counts = Counter(r.get("judgement", {}).get("verdict", "error" if r.get("error") else "unjudged") for r in rows)
    by_status = {}
    for status in ("reviewed", "pending", "disputed"):
        selected = [r for r in rows if r["review_status"] == status]
        by_status[status] = {
            "cases": len(selected),
            "generated": sum("draft_answer" in r for r in selected),
            "errors": sum("error" in r for r in selected),
            "verdicts": dict(Counter(r.get("judgement", {}).get("verdict", "error" if r.get("error") else "unjudged") for r in selected)),
        }
    summary = {
        **protocol,
        "actual_cases": len(rows),
        "generation_calls": sum("draft_answer" in r for r in rows),
        "judge_calls": sum("judgement" in r for r in rows),
        "verdicts": dict(counts),
        "by_review_status": by_status,
        "citation_format_valid": sum(bool(r.get("citation_check", {}).get("structurally_valid")) for r in rows),
        "errors": [{"id": r["id"], "error": r["error"]} for r in rows if r.get("error")],
    }
    (OUT / "api_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("API_COMPLETE", json.dumps({k: summary[k] for k in ("actual_cases", "generation_calls", "judge_calls", "verdicts", "by_review_status", "citation_format_valid", "errors")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
