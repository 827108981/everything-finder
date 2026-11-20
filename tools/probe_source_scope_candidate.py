"""Freeze and inspect a small source-scope sample before changing selection."""

from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.generation.base import TextCompletionRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

from tools.rag_eval_common import ROOT, read, sha, write
from tools.rag_trace_audit import original_text
from tools.source_scope_candidate import PROMPT, validate


OUT = ROOT / "reports/rag_general_branch_scope_20261111_01"
PILOT = ROOT / "reports/rag_conditional_fusion_pilot_20261110_01/runs.jsonl"
PLANS = ROOT / "reports/rag_conditional_fusion_cohort_20261110_01/plans.json"
SELECTION = (("rag-biz-008", "R", 1, ("S1", "S3", "S4")),
             ("rag-biz-031", "R", 1, ("S1",)),
             ("rag-biz-040", "R", 1, ("S6", "S7")))


def main():
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    secret = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    if not secret:
        raise RuntimeError("saved model credential unavailable")
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    rows = [json.loads(line) for line in PILOT.open(encoding="utf-8") if line.strip()]
    questions = {row["id"]: row["question"] for row in read(PLANS)}
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    write(OUT / "scope_probe_protocol.json", {
        "bundle_id": bundle.bundle_id, "source_run_sha256": sha(PILOT),
        "question_source_sha256": sha(PLANS),
        "settings_sha256": sha(SETTINGS_PATH),
        "scope_code_sha256": sha(ROOT / "tools/source_scope_candidate.py"),
        "selection": SELECTION, "input": "question and actual included_text only; no Gold or filename",
    })
    path = OUT / "scope_probe.jsonl"
    completed = set()
    if path.exists():
        completed = {(row["id"], row["source_id"]) for line in path.open(encoding="utf-8")
                     if (row := json.loads(line))}
    with store.connect_read_only(bundle.metadata_path) as connection, path.open(
        "a", encoding="utf-8"
    ) as result:
        for case_id, arm, repeat, source_ids in SELECTION:
            run = next(row for row in rows if row["id"] == case_id
                       and row["arm"] == arm and row["repeat"] == repeat)
            for source_id in source_ids:
                if (case_id, source_id) in completed:
                    continue
                entry = next(row for row in run["entries"] if row["source_id"] == source_id)
                question = questions[case_id]
                prompt = "用户问题：\n" + question + "\n\n资料正文：\n" + entry["included_text"]
                data = {"id": case_id, "source_id": source_id, "chunk_id": entry["chunk_id"]}
                try:
                    raw = gateway.complete_text(TextCompletionRequest(
                        system_prompt=PROMPT, user_prompt=prompt,
                        model=settings.rag_api_model, api_base_url=settings.rag_api_base_url,
                        api_style=settings.rag_api_style, api_key=secret,
                        timeout_seconds=settings.rag_api_timeout_seconds,
                        max_output_tokens=650,
                    ), CancelToken())
                    scope = validate(question, entry["included_text"], raw)
                    chunk = store.load_chunk(connection, entry["chunk_id"])
                    assert chunk is not None
                    original = original_text(Path(chunk.source.file_path),
                                             chunk.source.page_number or chunk.source.slide_number)
                    data.update(raw=raw, validated=vars(scope),
                                original_support_quote=bool(scope.support_quote and
                                    canonical_content(scope.support_quote) in canonical_content(original)),
                                original_scope_quote=bool(scope.scope_quote and
                                    canonical_content(scope.scope_quote) in canonical_content(original)))
                except Exception as exc:
                    data.update(error=str(exc).replace(secret, "[REDACTED]")[:500])
                result.write(json.dumps(data, ensure_ascii=False) + "\n")
                result.flush()
                print(case_id, source_id,
                      data.get("validated", {}).get("relation", data.get("error", "error")), flush=True)


if __name__ == "__main__":
    main()
