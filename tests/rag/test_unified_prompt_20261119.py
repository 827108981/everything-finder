from dataclasses import replace

import pytest

from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import (
    SYSTEM_PROMPT, UNIFIED_EVIDENCE_PROMPT, generation_system_prompt, provider_request_payload,
)


@pytest.mark.parametrize('style', ['chat_completions', 'responses'])
def test_unified_changes_only_system_and_keeps_default(style):
    request = GenerationRequest('普通问题', '[S1]\n原文含否定与单位', 'fixed-model',
                                'https://example.test/v1', style, 'private-key',
                                query_plan_summary='原问题摘要')
    base = provider_request_payload(request)
    new = provider_request_payload(replace(request, prompt_variant='unified'))
    assert request.prompt_variant == 'baseline'
    assert generation_system_prompt() == SYSTEM_PROMPT
    assert generation_system_prompt('unified') == UNIFIED_EVIDENCE_PROMPT
    if style == 'chat_completions':
        assert new['messages'][0]['content'] == UNIFIED_EVIDENCE_PROMPT
        assert new['messages'][1:] == base['messages'][1:]
        new['messages'][0] = base['messages'][0]
    else:
        assert new['instructions'] == UNIFIED_EVIDENCE_PROMPT
        new['instructions'] = base['instructions']
    assert new == base
    assert 'private-key' not in str(base)


def test_unified_has_no_gold_constants_or_few_shot_answers():
    assert not any(x in UNIFIED_EVIDENCE_PROMPT for x in
                   ('rag-biz', 'CSDB', 'LVC', 'BM10', '22.8', 'CL-8000', '115±5', '66000', 'E:\\'))
    assert '虚构例' not in UNIFIED_EVIDENCE_PROMPT
