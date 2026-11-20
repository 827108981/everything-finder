"""Compatibility at the real service entry; network/model providers are test doubles."""
from dataclasses import replace
import pytest

pytest.importorskip('chromadb')
from tests.rag.test_rag_end_to_end import indexed_rag, _Embedding, _Reranker, _Gateway
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.secrets import MemorySecretStore
from local_full_text_search.core.task_manager import CancelToken


@pytest.mark.parametrize('config',['', 'chemiluminescence', 'Z:/missing/config.json'])
def test_enhanced_plan_can_answer_generic_index_question_without_domain(config,indexed_rag):
    source,store,settings=indexed_rag
    settings=replace(settings,rag_query_plan_enabled=True,rag_domain_config=config)
    secrets=MemorySecretStore();secrets.save(settings.rag_api_secret_ref,'test-secret')
    gateway=_Gateway();events=[]
    service=AnswerService(source,store,_Embedding(),_Reranker(.97),gateway,secrets,settings,progress_callback=events.append)
    answer=service.answer(RagQuery('索引任务中断后如何恢复？'),CancelToken())
    assert not answer.insufficient_evidence and gateway.calls==1
    snapshot=next(e['query_plan'] for e in events if 'query_plan' in e)
    assert snapshot['analyzer_version']=='query-plan-v3'
    assert not snapshot['adapter_active']
    assert all(isinstance(e,str) for e in snapshot['entities'])
    assert 'test-secret' not in str(events)
