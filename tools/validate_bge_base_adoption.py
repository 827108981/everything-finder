"""Verify the default BGE strategy against frozen B and actual retrieve, no API."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['ANONYMIZED_TELEMETRY']='False'
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
from pathlib import Path
from tools.validate_rerank_semantic import OUT as HISTORY, read_jsonl,Token
from tools.validate_selection_factorial import ROOT,load_inputs,reconstruct,no_network,sha,write,score_stage
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.index_store import RagIndexStore

OUT=ROOT/'reports/rag_bge_base_adoption_20261116'


class Forbidden:
    def retrieve(self,*a,**k):raise AssertionError('Credentials forbidden')
    def generate(self,*a,**k):raise AssertionError('Generation forbidden')


def replay():
    rows,settings,_,mappings=load_inputs()
    protocol=json.loads((HISTORY/'protocol.json').read_text(encoding='utf8'))
    assert all(sha(p)==h for p,h in protocol['protected'].items())
    expected={r['id']:r for r in read_jsonl(HISTORY/'B.jsonl')}
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    assert bundle.bundle_id==protocol['bundle']
    service=AnswerService(None,store,None,None,None,None,AppSettings.from_dict(settings))
    ranker=object.__new__(LocalBgeReranker)
    checks=[];output=[]
    with no_network() as attempts,store.connect_read_only(bundle.metadata_path) as conn:
        for row in rows:
            plan,items=reconstruct(row,store,conn)
            lookup={c.chunk.stable_chunk_id:c for c in items}
            items=[lookup[c['chunk_id']] for c in row['candidate_stages']['rerank']]
            for c in items:c.constraint_decision=None;c.constraint_score=0
            items=ranker._order_scored(row['question'],items,Token())
            rerank=[candidate_snapshot(c) for c in items]
            items=ConstraintMatcher().evaluate(plan,items)
            conditioned=[candidate_snapshot(c) for c in items]
            d=service.evidence_policy.evaluate(RagQuery(row['question'],final_top_k=settings['rag_final_top_k']),items,plan=plan)
            ctx=service.context_builder.build(d.accepted,max_chars=settings['rag_context_max_chars'])
            e=expected[row['id']]
            checks.append(dict(id=row['id'],order_equal=[c['chunk_id'] for c in rerank]==[c['chunk_id'] for c in e['rerank_candidates']],
                condition_order_equal=[c['chunk_id'] for c in conditioned]==[c['chunk_id'] for c in e['condition_candidates']],
                context_equal=ctx.text==e['context'],entries_equal=list(ctx.entries)==e['entries'],
                score_equal=score_stage('context',list(ctx.entries),mappings[row['id']])==e['scores']['context']))
            record=deepcopy(e)
            record.update(rerank_candidates=rerank,condition_candidates=conditioned,context=ctx.text,entries=list(ctx.entries),
                selection_trace=[dict(rank=i,chunk_id=c['chunk_id'],raw_ce=c['rerank_score'],diagnostics=c['rerank_diagnostics']) for i,c in enumerate(rerank,1)])
            output.append(record)
    assert all(all(v for k,v in c.items() if k!='id') for c in checks)
    (OUT/'B.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf8')
    (OUT/'baseline.jsonl').write_bytes((HISTORY/'baseline.jsonl').read_bytes())
    write(OUT/'production_replay.json',dict(passed=True,cases=91,checks=checks,api_calls=0,network_attempts=attempts,
        method='production _order_scored, fixed original CE scores; production conditions/evidence/Context',
        protected_files_unchanged=all(sha(p)==h for p,h in protocol['protected'].items()),bundle=bundle.bundle_id))
    print('Production B matches frozen B:91/91',flush=True)


def native():
    import torch
    torch.set_num_threads(3)
    from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
    from local_full_text_search.rag.model_manager import LocalModelManager,ModelSpec
    from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
    from local_full_text_search.core.index_generation_store import IndexGenerationStore
    from local_full_text_search.core.task_manager import CancelToken
    from tools.validate_query_plan_adapter import STAGES
    rows,s,_,_=load_inputs();settings=AppSettings.from_dict(s)
    wanted={'rag-biz-002','rag-biz-004','rag-biz-006','rag-biz-021','rag-row-holdout2-001'}
    expected={r['id']:r for r in read_jsonl(OUT/'B.jsonl')}
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR);records=[];checks=[]
    with no_network() as attempts:
        embed=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),
            repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,batch_size=settings.rag_embedding_batch_size)
        ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),
            repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length)
        assert ranker.ordering_policy is None
        for row in rows:
            if row['id'] not in wanted:continue
            events=[];stages={}
            def progress(e):
                events.append(e)
                for field,stage in STAGES.items():
                    if field in e:stages[stage]=e[field]
            service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),RagIndexStore('E:/everythingfinder-db/rag'),
                embed,ranker,Forbidden(),Forbidden(),settings,progress_callback=progress)
            q=RagQuery(row['question'],include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,lexical_top_k=settings.rag_lexical_top_k,
                vector_top_k=settings.rag_vector_top_k,fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=settings.rag_final_top_k)
            result=service.retrieve(q,CancelToken())
            ctx=next(e for e in reversed(events) if 'llm_context' in e)
            e=expected[row['id']]
            checks.append(dict(id=row['id'],context_equal=ctx['llm_context']==e['context'],entries_equal=ctx['context_entries']==e['entries'],
                rerank_equal=[c['chunk_id'] for c in stages['rerank']]==[c['chunk_id'] for c in e['rerank_candidates']]))
            records.append(dict(id=row['id'],question=row['question'],result=asdict(result),events=events,stages=stages))
            (OUT/'native.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,default=str)+'\n' for r in records),encoding='utf8')
            print('Native B',row['id'],checks[-1],flush=True)
    assert len(checks)==5 and all(all(v for k,v in c.items() if k!='id') for c in checks) and not attempts
    write(OUT/'native_audit.json',dict(passed=True,checks=checks,network_attempts=attempts,api_calls=0,
        real_models=True,ordering_policy=None,adapter_enabled=settings.rag_query_plan_enabled))


def report():
    from tools.render_selection_factorial import main as render
    from tools.analyze_rerank_semantic import summarize
    metrics={}
    for arm in ('baseline','B'):
        records=read_jsonl(OUT/(arm+'.jsonl'))
        metrics[arm]={cohort:summarize([r for r in records if r['cohort']==cohort]) for cohort in ('formal','diagnostic')}
    write(OUT/'metrics.json',metrics)
    # These source-level findings remain applicable: actual Context equals frozen B.
    write(OUT/'negative_review.json',json.loads((HISTORY/'negative_review.json').read_text(encoding='utf8')))
    banner='''<section class="panel"><h2>已采用BGE基础排序：旧P与当前B对照</h2>
    <p>按用户指令切换默认重排。移除辅助分及融合旧名次保护；结构规则与本地4B模型未启用。API调用0。
    冻结91题与原20候选验证当前排序、条件顺序、实际Context同实验B一致。正式全部Gold覆盖41/46→43/46；诊断35/40不变。</p>
    <p>004/006恢复；019入口缺页、条码条件排序、错平台资料干扰仍按原记录保留，未声称已解决。</p>
    <label>切换方案：<select id="armSelect"><option value="baseline">旧P</option><option value="B" selected>当前BGE基础排序</option></select></label>
    <p id="sourceStatus">通过同目录open_report.cmd打开，可点击定位原资料页。</p><a href="conclusion.md">本轮接入结论</a></section>'''
    render(output=OUT,arms={a:SimpleNamespace(remove_document_cap=True,budget_pack=False) for a in ('baseline','B')},
        labels={'baseline':'旧P混合分+旧保护','B':'当前BGE基础排序'},reader=lambda a:{r['id']:r for r in read_jsonl(OUT/(a+'.jsonl'))},
        mutable_stages=('rerank','constraints','evidence','context'),initial_arm='B',banner_override=banner)


def audit():
    import xml.etree.ElementTree as ET
    old=json.loads((HISTORY/'final_acceptance.json').read_text(encoding='utf8'))
    tests=ET.parse(OUT/'pytest.xml').findall('.//testcase')
    failed=[(c.get('classname'),c.get('name')) for c in tests if c.find('failure') is not None]
    assert sorted(failed)==sorted(tuple(c) for c in old['unit_tests']['existing_failures'])
    protocol=json.loads((HISTORY/'protocol.json').read_text(encoding='utf8'))
    drift=[p for p,h in protocol['protected'].items() if sha(p)!=h]
    assert not drift
    replay_check=json.loads((OUT/'production_replay.json').read_text(encoding='utf8'))
    native_check=json.loads((OUT/'native_audit.json').read_text(encoding='utf8'))
    ui=json.loads((OUT/'report_ui_audit.json').read_text(encoding='utf8'))
    assert replay_check['passed'] and native_check['passed'] and ui['verified_arms']==['baseline','B'] and not ui['page_errors']
    counts=dict(passed=sum(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in tests),
                failed=len(failed),skipped=sum(c.find('skipped') is not None for c in tests))
    # JUnit represents xfail as skipped; keep pytest's human summary too.
    result=dict(adopted='BGE base',user_authorized=True,default_changed=True,
        effective_weights={'cross_encoder':1,'retrieval_prior':0,'answer_support':0,'max_rank_drop':0},
        optional_ordering_policy=None,structure_rules_enabled=False,local_semantic_model_enabled=False,
        api_calls=0,bundle=protocol['bundle'],protected_drift=drift,persistent_settings_changed=False,
        gold_changed=False,replay=replay_check,native=native_check,ui=ui,tests=counts,
        pytest_summary=(OUT/'pytest.log').read_text(encoding='utf8').splitlines()[-1],
        existing_failures=failed,new_failures=[],all_green=False,
        remaining_problems='Known scope contamination and condition-ordering/entry losses unchanged; adoption is not a claim these are fixed')
    write(OUT/'final_acceptance.json',result)
    (OUT/'conclusion.md').write_bytes((ROOT/'docs/rag_bge_base_adoption_20261116.md').read_bytes())
    print('BGE default adopted: replay91/91, native5/5, API0, no new failures, protected drift0',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['replay','native','report','audit'],required=True);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    {'replay':replay,'native':native,'report':report,'audit':audit}[args.phase]()


if __name__=='__main__':main()
