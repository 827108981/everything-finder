"""Real AnswerService.answer with real local retrieval and existing application settings."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from tools.validate_prompt_factorial_20261119 import OUT, ROOT, TARGETS, check_frozen, inputs, write
from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway, provider_request_payload
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from tools.validate_vector_packaging_20261116 import model


def main():
    protocol = check_frozen()
    cases, contexts, settings = inputs()
    output = OUT / 'service_entry.jsonl'
    prior = [json.loads(l) for l in output.read_text(encoding='utf8').splitlines()] if output.exists() else []
    done = {(r['id'],r['arm']) for r in prior}
    store = RagIndexStore('E:/everythingfinder-db/rag')
    bundle = store.active()
    assert bundle.bundle_id == protocol['bundle_id']
    index = VectorIndexStore(store.root).active(bundle.bundle_id, embedding_repo_id=bundle.embedding_repo_id,
        embedding_revision=bundle.embedding_revision, embedding_dimension=bundle.embedding_dimension,
        policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint)
    class BoundStore:
        def __init__(self, *args, **kwargs): pass
        def attached(self, source_bundle_id, *args, **kwargs):
            assert source_bundle_id == index.source_bundle_id
            return index
        active = attached
    key_store = WindowsCredentialSecretStore()
    key = key_store.retrieve(settings.rag_api_secret_ref)
    assert key
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    judge = GoldAnswerJudge(gateway, settings, key)
    class PromptGateway:
        def __init__(self, arm): self.arm, self.requests = arm, []
        def generate(self, request, token):
            effective = replace(request, prompt_variant=self.arm)
            self.requests.append(provider_request_payload(effective))
            return gateway.generate(effective, token)
        def assess_clarification(self, *args): return gateway.assess_clarification(*args)
        def review(self, *args): return gateway.review(*args)
    emb = model(bundle)
    path = LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,
        settings.rag_reranker_revision, 'reranker'))
    ranker = LocalBgeReranker(path, repo_id=settings.rag_reranker_repo_id,
        revision=settings.rag_reranker_revision, batch_size=4, max_length=settings.rag_reranker_max_length)
    assert ranker.input_mode == 'window_max'
    write(OUT / 'service_entry_protocol.json', dict(cases=list(TARGETS), arms=['baseline','combined'],
        actual_local_retrieval=True, bundle_id=bundle.bundle_id, metadata_sha256=sha256(Path(bundle.metadata_path).read_bytes()).hexdigest(),
        adapter=False, semantic_clarification_enabled=settings.rag_semantic_clarification_enabled,
        answer_review_enabled=settings.rag_answer_review_enabled,
        note='Prompt gateway changes only the generation system instructions. Existing enabled/disabled prechecks and postchecks run through AnswerService.answer; no Gold in runtime. Trace generating payload is baseline construction, effective override is recorded explicitly.'))
    with patch('local_full_text_search.rag.answer_service.VectorIndexStore', BoundStore):
        for case_id in TARGETS:
            for arm in ('baseline','combined'):
                if (case_id,arm) in done: continue
                events = []
                prompt_gateway = PromptGateway(arm)
                service = AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')), store,
                    emb, ranker, prompt_gateway, key_store, settings, progress_callback=events.append)
                record = dict(id=case_id, arm=arm, entry='AnswerService.answer', model=settings.rag_api_model)
                try:
                    result = service.answer(RagQuery(cases[case_id].question, include_ocr=True,
                        min_ocr_confidence=settings.ocr_min_confidence, lexical_top_k=50, vector_top_k=50,
                        fusion_top_k=30, reranker_top_k=20, final_top_k=8), CancelToken())
                    ctx = next((e for e in reversed(events) if 'llm_context' in e),{})
                    actual = ctx.get('llm_context','')
                    record.update(result=asdict(result), context=actual, entries=ctx.get('context_entries',[]),
                        events=events, effective_generation_prompt_variant=arm,
                        effective_api_requests=prompt_gateway.requests,
                        context_equal_frozen=actual==contexts[case_id]['context'],
                        judgement=judge.evaluate(cases[case_id],dict(status=result.status,text=result.answer_text,
                            evidence_context=actual),CancelToken()))
                except Exception as exc:
                    record.update(error=str(exc).replace(key,'[REDACTED]')[:600],events=events)
                with output.open('a',encoding='utf8') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
                print('SERVICE',case_id,arm,record.get('context_equal_frozen'),record.get('judgement',{}).get('verdict'),record.get('error',''),flush=True)
    check_frozen()


if __name__=='__main__': main()
