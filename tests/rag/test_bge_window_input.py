from types import MethodType
import math
from local_full_text_search.core.errors import CancelledError

import pytest

from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.errors import RagModelValidationError
from local_full_text_search.core.task_manager import CancelToken
from tests.rag.test_rerank_support import _candidate


@pytest.fixture
def ranker():
    tokenizers = pytest.importorskip('tokenizers')
    transformers = pytest.importorskip('transformers')
    vocab = {s:i for i,s in enumerate(['<unk>','<s>','</s>','<pad>','query',*[f'w{i}' for i in range(90)]])}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single='<s> $A </s>', pair='<s> $A </s> </s> $B </s>',
        special_tokens=[('<s>',vocab['<s>']),('</s>',vocab['</s>'])])
    t = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='<pad>',unk_token='<unk>',model_max_length=32)
    t.model_input_names=['input_ids','attention_mask']
    r=object.__new__(LocalBgeReranker)
    r.tokenizer=t;r.max_length=32;r.window_overlap=4;r.input_mode='window_max';r.ordering_policy=None
    def score(self,inputs,token):
        return [(p,math.log(p/(1-p))) for e in inputs for p in [.95 if vocab['w69'] in e['input_ids'] else .05]]
    r._score_window_inputs=MethodType(score,r)
    return r


def test_windows_cover_all_original_tokens_and_preserve_question(ranker):
    text=' '.join(f'w{i}' for i in range(70))
    windows=ranker._windows('query',text)
    assert len(windows)>1
    assert windows[0][1]['start_char']==0
    assert windows[-1][1]['end_char']==len(text)
    for (e,m),(n,nm) in zip(windows,windows[1:]):
        assert nm['start_char']<m['end_char']
        assert e['input_ids'][:3]==n['input_ids'][:3]
    assert all(m['tokens']<=32 for _,m in windows)
    assert all(any(m['start_char']<=text.index(f'w{i}') and text.index(f'w{i}')+len(f'w{i}')<=m['end_char'] for _,m in windows) for i in range(70))


def test_short_input_token_identical_to_legacy(ranker):
    expected=ranker.tokenizer('query','w1 w2 w3',truncation=True,max_length=32)
    windows=ranker._windows('query','w1 w2 w3')
    assert len(windows)==1
    assert windows[0][0]==dict(expected)


def test_late_evidence_affects_parent_score_without_changing_source(ranker):
    text=' '.join(f'w{i}' for i in range(70))
    item=_candidate(text,'long')
    before=item.chunk
    ranked=ranker.rerank('query',[item],20,CancelToken())
    assert ranked[0].rerank_score==.95
    assert item.chunk==before
    assert item.rerank_diagnostics['window_count']>1
    assert item.rerank_diagnostics['windows'][-1]['end_char']==len(text)
    assert item.rerank_diagnostics['strategy']=='bge_base'
    ranker.input_mode='window_top2_mean'
    ranker.rerank('query',[item],20,CancelToken())
    assert item.rerank_score==.5
    assert item.rerank_diagnostics['logit_kind']=='aggregate_probability_logit'


def test_single_window_not_penalized_by_top2_mean(ranker):
    item=_candidate('w69','short')
    ranker.input_mode='window_top2_mean'
    ranker.rerank('query',[item],20,CancelToken())
    assert item.rerank_score==.95
    assert item.rerank_diagnostics['logit_kind']=='model_logit'


def test_long_question_is_not_silently_truncated(ranker):
    with pytest.raises(RagModelValidationError):
        ranker._windows(' '.join(['query']*32),'w1')


def test_tiny_remaining_budget_clamps_overlap(ranker):
    windows=ranker._windows(' '.join(['query']*26),'w1 w2 w3 w4')
    assert windows[-1][1]['end_char']==len('w1 w2 w3 w4')
    assert all(m['tokens']<=32 for _,m in windows)


def test_window_input_obeys_cancellation(ranker):
    token=CancelToken();token.cancel()
    with pytest.raises(CancelledError):
        ranker.rerank('query',[_candidate('w1','a')],20,token)


def test_default_constructor_reads_tail_with_max_aggregation(ranker, monkeypatch):
    import transformers
    from types import SimpleNamespace
    class Model:
        config=SimpleNamespace(num_labels=1)
        def to(self, device):return self
        def eval(self):pass
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:ranker.tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification,'from_pretrained',lambda *a,**k:Model())
    adopted=LocalBgeReranker(None,repo_id='fixture',revision='fixed',max_length=32,
                           cross_encoder_weight=.65,answer_support_weight=.30,max_rank_drop=3)
    assert adopted.input_mode=='window_max'
    assert adopted.window_overlap==64
    adopted._score_window_inputs=MethodType(ranker._score_window_inputs.__func__,adopted)
    item=_candidate(' '.join(f'w{i}' for i in range(70)),'tail')
    adopted.rerank('query',[item],20,CancelToken())
    assert item.rerank_score==.95
    assert item.rerank_diagnostics['window_count']>1
    assert adopted.answer_support_weight==0 and adopted.max_rank_drop==0
