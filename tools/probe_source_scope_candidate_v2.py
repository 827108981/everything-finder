"""Branch-level scope probe with a frozen source and branch matrix."""

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

from tools.branch_candidate_v2 import validate as validate_branches
from tools.rag_eval_common import ROOT, read, sha, write
from tools.rag_trace_audit import original_text
from tools.source_scope_candidate_v2 import PROMPT, validate


OUT = ROOT / "reports/rag_general_branch_scope_20261111_01"
PILOT = ROOT / "reports/rag_conditional_fusion_pilot_20261110_01/runs.jsonl"
PLANS = ROOT / "reports/rag_conditional_fusion_cohort_20261110_01/plans.json"
MATRIX = (("rag-biz-008", "S1", (1, 2)),
          ("rag-biz-008", "S3", (1, 2)),
          ("rag-biz-008", "S4", (2,)),
          ("rag-biz-031", "S1", (1, 2)),
          ("rag-biz-040", "S6", (1, 2)),
          ("rag-biz-040", "S7", (1, 2)))


def main():
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    secret = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    if not secret:
        raise RuntimeError("saved model credential unavailable")
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    plans = {row["id"]: row for row in read(PLANS)}
    runs = [json.loads(line) for line in PILOT.open(encoding="utf-8") if line.strip()]
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    write(OUT / "scope_probe_v2_protocol.json", {
        "bundle_id": bundle.bundle_id, "run_sha256": sha(PILOT),
        "plans_sha256": sha(PLANS), "settings_sha256": sha(SETTINGS_PATH),
        "code_sha256": sha(ROOT / "tools/source_scope_candidate_v2.py"),
        "matrix": MATRIX, "revision_reason": "whole-chunk first probe mixed branch scope and cause",
        "input": "question, one branch and actual included_text; no Gold or filename",
    })
    path = OUT / "scope_probe_v2.jsonl"
    completed = set()
    if path.exists():
        completed = {(row["id"], row["source_id"], row["branch"])
                     for line in path.open(encoding="utf-8") if (row := json.loads(line))}
    with store.connect_read_only(bundle.metadata_path) as connection, path.open(
        "a", encoding="utf-8"
    ) as output:
        for case_id, source_id, branch_ids in MATRIX:
            run = next(row for row in runs if row["id"] == case_id
                       and row["arm"] == "R" and row["repeat"] == 1)
            plan = plans[case_id]
            branches = validate_branches(plan["question"], plan["raw"])
            entry = next(row for row in run["entries"] if row["source_id"] == source_id)
            for branch_id in branch_ids:
                if (case_id, source_id, branch_id) in completed:
                    continue
                branch = branches[branch_id - 1]
                prompt = ("完整用户问题：\n" + plan["question"] + "\n\n指定分支：\n"
                          + branch.condition_quote + "\n共享对象或指标：\n"
                          + (branch.inherited_quote or "无") + "\n\n资料正文：\n"
                          + entry["included_text"])
                data = {"id": case_id, "source_id": source_id, "branch": branch_id,
                        "chunk_id": entry["chunk_id"]}
                try:
                    raw = gateway.complete_text(TextCompletionRequest(
                        system_prompt=PROMPT, user_prompt=prompt,
                        model=settings.rag_api_model, api_base_url=settings.rag_api_base_url,
                        api_style=settings.rag_api_style, api_key=secret,
                        timeout_seconds=settings.rag_api_timeout_seconds,
                        max_output_tokens=650,
                    ), CancelToken())
                    record = validate(plan["question"], entry["included_text"], raw)
                    chunk = store.load_chunk(connection, entry["chunk_id"])
                    assert chunk is not None
                    original = original_text(Path(chunk.source.file_path),
                                             chunk.source.page_number or chunk.source.slide_number)
                    data.update(raw=raw, validated=vars(record),
                                original_support_quote=bool(record.support_quote and
                                    canonical_content(record.support_quote) in canonical_content(original)),
                                original_scope_quote=bool(record.scope_quote and
                                    canonical_content(record.scope_quote) in canonical_content(original)))
                except Exception as exc:
                    data.update(error=str(exc).replace(secret, "[REDACTED]")[:500])
                output.write(json.dumps(data, ensure_ascii=False) + "\n")
                output.flush()
                print(case_id, source_id, branch_id,
                      data.get("validated", {}).get("relation", data.get("error", "error")), flush=True)


if __name__ == "__main__":
    main()
