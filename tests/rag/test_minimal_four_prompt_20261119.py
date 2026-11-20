from dataclasses import replace

import pytest

from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import (
    SYSTEM_PROMPT, MINIMAL_FOUR_PROMPT, generation_system_prompt, provider_request_payload,
)


@pytest.mark.parametrize('style', ['chat_completions', 'responses'])
def test_minimal_four_only_changes_system(style):
    request = GenerationRequest('问题原话', '[S1]\n原资料', 'test-model',
        'https://example.test/v1', style, 'test-secret', query_plan_summary='辅助摘要')
    baseline = provider_request_payload(request)
    candidate = provider_request_payload(replace(request, prompt_variant='minimal_four'))
    if style == 'chat_completions':
        assert candidate['messages'][1:] == baseline['messages'][1:]
        candidate['messages'][0] = baseline['messages'][0]
    else:
        candidate['instructions'] = baseline['instructions']
    assert candidate == baseline
    assert generation_system_prompt() == SYSTEM_PROMPT
    assert request.prompt_variant == 'baseline'


def test_minimal_four_preserves_boundaries_and_refusal_literal():
    text = generation_system_prompt('minimal_four')
    original_lines = SYSTEM_PROMPT.splitlines()
    assert all(line in text for line in original_lines[:-1])
    assert '请只回答“资料中没有足够依据。”。' in text
    assert '本次提供的资料中没有足够依据' not in text
    assert text.endswith(MINIMAL_FOUR_PROMPT)
    assert len(MINIMAL_FOUR_PROMPT.splitlines()) == 4
    assert not any(x in text for x in ('rag-biz', 'CSDB', 'LVC', '22.8', '66000', 'BM10', 'E:\\'))
