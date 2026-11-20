"""Apply human-authorized question/source corrections as a new Gold revision.

This is annotation tooling, never imported by production retrieval/generation.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json

from tools.validate_selection_factorial import ROOT, sha, write
from tools.diagnose_vector_recall import ro
from tools.rag_trace_audit import original_text
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence, load_chunk_rows
from local_full_text_search.rag.evaluation.stage_scoring import score_stage

F=ROOT/'tests/rag/fixtures'
OUT=ROOT/'reports/rag_gold_material_repair_20261116'
VERSION='business-gold-v5.1-20261116'
NAMES={'rag_gold_v5':'rag_gold_v5_1','rag_row_holdout_v1_1':'rag_row_holdout_v1_2','rag_independent_v2':'rag_independent_v2_1'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((F/'rag_active_suite_v5.json').read_text(encoding='utf8'))
    protected={str(F/d['path'].split('/')[-1]):sha(ROOT/d['path']) for d in manifest['datasets']}
    protocol=json.loads((ROOT/'reports/rag_vector_diagnosis_20261116/protocol.json').read_text(encoding='utf8'))
    protected.update({p:h for p,h in protocol['protected'].items() if not p.startswith('tests')})
    docs={name:json.loads((F/(name+'_review.json')).read_text(encoding='utf8')) for name in NAMES}
    old={c['id']:deepcopy(c) for doc in docs.values() for c in doc['cases']}
    cases={c['id']:c for doc in docs.values() for c in doc['cases']}
    with ro('E:/everythingfinder-db/rag/rag.g000033-71ec3c52f3/metadata.db') as conn:
        indexed=load_chunk_rows(conn)
    verifications=[]
    def verify(path,page,quote):
        actual=original_text(Path(path),int(page))
        rows=[r for r in indexed if Path(r['file_path'])==Path(path) and (r.get('slide_number') or r.get('page_number'))==int(page)]
        assert canonical_content(quote) in canonical_content(actual),(path,page,quote)
        assert any(canonical_content(quote) in canonical_content(r['text']) for r in rows),(path,page,quote)
        verifications.append(dict(file=str(path),page=int(page),quote=quote,original_verified=True,index_verified=True,source_sha256=sha(Path(path))))
    # Authoritative primary quote describes the fact, not the table's arbitrary row number.
    cases['rag-row-holdout-005']['gold_evidence'][0]['quote']='电极斜率范围\nNa+/K+: 45-70mv/dec, Cl-: 40-70mv/dec,'
    approved=json.loads((ROOT/'reports/rag_vector_diagnosis_20261116/alternative_evidence_review.json').read_text(encoding='utf8'))
    for alt in approved:
        c=cases[alt['id']]
        for entry in c['gold_evidence']:
            q=entry['quote']
            if not any(canonical_content(q)==canonical_content(a) for a in alt['quote']):continue
            verify(alt['file'],alt['page'],q)
            entry.setdefault('alternative_sources',[]).append(dict(
                file=alt['file'],page=alt['page'],position='原表头和同事实对应行/暗计数诊断原因列表',
                quote=q,relevance=2,approved=True,approved_by='user',approval_date='2026-11-16',
                approval_basis='用户明确要求将已核实的正确替代资料加入Gold；同一事实OR匹配，不增加必答项。',
                why_sufficient=alt['scope_review'].replace('未自行登记正式Gold替代。','仅本题核实事实适用。'),source_sha256=sha(Path(alt['file']))))
    def update(cid,q,facts,forbidden,obj):
        c=cases[cid]
        c.update(question=q,expected_behavior='answer',required_facts=facts,
                 answer_checks=[dict(id=f'F{i}',role='core',text=t) for i,t in enumerate(facts,1)],
                 forbidden_claims=forbidden,missing_fields=[])
        c['constraints']['object']=obj
        c['validation_category']='answer'
        return c
    c=update('rag-biz-028','发光临床结果普遍偏低，该从哪些方面排查？',[
        '检查试剂是否超过定标周期、底物液是否过期。','检查校准品是否失效、使用操作是否正确。','检查仪器是否长期未维护保养。'],
        ['未进行实际检查就把某一种可能原因说成已经确认的故障。'],'发光临床结果')
    # The second, calibration-oriented generality existed to support the removed
    # board inference question. It is no longer a mandatory retrieval target.
    c['gold_evidence']=c['gold_evidence'][:1]
    c['gold_evidence'][0]['why_sufficient']='直接列出临床系统性偏高/偏低的常见原因，支持本题三个排查方向。'
    c['hard_negatives']=[]
    c['group_id']='systematic_clinical_result_low_limits' # stable business group, no new independent sample
    c=update('evidence-flow-independent-06','BS-2800M 和 BS-430 的分析部与电脑分别通过什么接口通信？',[
        'BS-2800M分析部通过网口与电脑通信。','BS-430分析部通过串口与电脑通信。'],
        ['将BS-2800M与BS-430的通信接口互换。'],'BS-2800M/BS-430 分析部与电脑通信接口')
    c=update('evidence-flow-independent-05','A&T 和 Medica ISE 的测量原理有什么不同？如何判断这是哪种 ISE？',[
        'A&T采用间接离子选择电极法，样本先稀释再测量；Medica采用直接离子选择电极法。',
        '区分时核对模块类型与对应的检测原理、适用产品信息；是否先稀释可区分直接法与间接法。'],
        ['颠倒A&T与Medica的测量原理。','仅凭直接法就断定模块一定是Medica。','把某机型在不同模块或序列号条件下的适配说成唯一固定配置。'],
        'A&T/Medica ISE 测量原理与模块类型识别')
    # The question asks about A&T and Medica. A third manufacturer's example is
    # not an additional mandatory answer or retrieval target.
    c['identification_scope']='本题回答识别方法，不要求凭空判断一台未提供信息的现场仪器；不强制用户先补资料才解释。'
    changes=[]
    for cid,c in cases.items():
        if c!=old[cid]:
            c['annotation_revision']=dict(version=VERSION,date='2026-11-16',basis='用户明确修正问题意图及批准同事实替代来源；原review_status不变。')
            changes.append(dict(id=cid,before=old[cid],after=deepcopy(c)))
    for c in cases.values():
        if c['id'] not in {x['id'] for x in changes}:continue
        assert c['review_status']==old[c['id']]['review_status']
        for e in c['gold_evidence']:
            verify(e['file'],e['page'],e['quote'])
    new_manifest=deepcopy(manifest)
    new_manifest.update(version=VERSION,previous_manifest='tests/rag/fixtures/rag_active_suite_v5.json',
        comparison_policy='来源登记变更是评测口径变化；三道改问法题必须重新检索，不可把旧结果重标当新问法运行；不称为算法收益。')
    validation=[]
    for name,newname in NAMES.items():
        d=docs[name]
        d['_说明']=dict(version=VERSION,previous_file=f'tests/rag/fixtures/{name}_review.json',
            cases=len(d['cases']),review_policy='逐题审核状态不变；用户批准的替代来源仅对登记事实按OR计分。')
        d['_替代证据说明']='gold_evidence每项是一条必需证据；alternative_sources为同项的已批准替代来源，任选一处完整覆盖即可；不得把所有替代都要求召回。'
        write(F/(newname+'_review.json'),d)
        (F/(newname+'.jsonl')).write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in d['cases']),encoding='utf8')
        a=load_gold_dataset(F/(newname+'.jsonl'),reviewed_only=False)
        b=load_gold_dataset(F/(newname+'_review.json'),reviewed_only=False)
        assert [c.raw for c in a.cases]==[c.raw for c in b.cases]
        for item in new_manifest['datasets']:
            if item['path'].endswith(name+'.jsonl'):
                item.update(path=f'tests/rag/fixtures/{newname}.jsonl',sha256=sha(a.path),count=len(a.cases),status_counts=dict(Counter(c.review_status for c in a.cases)))
        validation.append(dict(file=str(a.path),count=len(a.cases),json_jsonl_equal=True,loader='pass'))
    main=docs['rag_gold_v5']['cases']
    reviewed=F/'rag_gold_v5_1_reviewed.jsonl'
    reviewed.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in main if c['review_status']=='reviewed'),encoding='utf8')
    assert len(load_gold_dataset(reviewed).cases)==47
    write(F/'rag_active_suite_v5_1.json',new_manifest)
    # Source-only score comparison reuses old query retrieval explicitly. New
    # question rows are marked stale, never called a fresh retrieval run.
    frozen={r['id']:r for r in map(json.loads,(ROOT/'reports/rag_support_graph_20261113/A.jsonl').read_text(encoding='utf8').splitlines())}
    scores=[]
    for cid in ['rag-biz-050','rag-row-holdout-004','rag-row-holdout-005','evidence-flow-independent-05']:
        c=next(c for name in NAMES.values() for c in load_gold_dataset(F/(name+'.jsonl'),reviewed_only=False).cases if c.case_id==cid)
        ms=map_case_evidence(c,indexed)
        assert all(m.alternatives for m in ms if m.kind=='positive')
        scores.append(dict(id=cid,question_changed=old[cid]['question']!=c.question,
            before=frozen[cid]['scores']['vector'],after_on_frozen_old_query=score_stage('vector',frozen[cid]['candidate_stages']['vector'],ms),
            interpretation='annotation-only comparison on old candidates; no algorithm gain; changed question must rerun'))
    write(OUT/'gold_changes.json',changes)
    write(OUT/'source_audit.json',verifications)
    write(OUT/'annotation_only_scores.json',scores)
    write(OUT/'validation.json',dict(total=sum(d['count'] for d in new_manifest['datasets']),changed_cases=len(changes),
        approved_alternative_sources=sum(len(e.get('alternative_sources',[])) for c in cases.values() for e in c['gold_evidence']),
        review_status_unchanged=True,files=validation,protected=protected,protected_drift=[p for p,h in protected.items() if sha(ROOT/p)!=h],api_calls=0))
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    print('Created v5.1; changed cases',len(changes),'source verifications',len(verifications),'historical fixtures/index/settings unchanged.')


if __name__=='__main__':main()
