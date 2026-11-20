"""Apply user-authorized four-case revisions, preserving all historical labels."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import json

from tools.rag_eval_common import ROOT, read, write, sha
from tools.rag_trace_audit import audit_case_sources
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows
from local_full_text_search.rag.index_store import RagIndexStore

F = ROOT / 'tests/rag/fixtures'
OUT = ROOT / 'reports/rag_gold_v4_3_retest_20261110'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = read(F / 'rag_gold_v4_2_review.json')
    doc = deepcopy(old)
    by = {c['id']: c for c in doc['cases']}
    edits = {
        '019': (None, [
            '排查仪器管路、分离液和底物液污染，以及加样系统异常等共性因素。',
            '先排除故障和污染源，再执行灌注。',
            '建议灌注40次以上；这是该处理思路中的建议次数。',
            '灌注后重新定标。',
        ]),
        '033': ('多个项目质控失控，有的显示“＞”、有的显示“＜”，应该怎样分步排查？', [
            '先核对哪些项目在控、哪些失控，以及同一复合质控内和不同复合质控的表现，判断异常范围。',
            '核对异常项目的方法学；若“＞”集中于竞争法、“＜”集中于夹心法，可提示发光值偏低。',
            '复测并观察失控项目发光值的变化；若发光值有升高趋势，应考虑磁珠未充分混匀。',
            '结合上述表现检查试剂磁珠是否充分混匀，不能仅凭“＞/＜”确定唯一故障原因。',
        ]),
        '040': ('ALB、GLU 等小样本量项目重复性变差，而 ALT、AST 较稳定；另一种情况是所有项目都变差。这两种现象分别提示优先检查哪个部件？', [
            '小样本量项目先变差、其他项目较稳定时，优先检查样本针是否轻微堵塞。',
            '所有项目都变差时，可优先检查冲洗头是否堵塞；这是可能的排查方向，不代表已经确认堵塞。',
        ]),
        '049': ('CL-6000i 的 SDM 非运动参数怎么备份？不同版本都能一键保存吗？', [
            'SDM3.0使用网口通讯，可以一键保存模块内全部单元参数。',
            'SDM1.0使用串口通讯，不能一键保存全部单元，需要逐个单元备份。',
            '串口备份时，在“非运动参数”中选择单元，点击“查询”，再点击“保存”并选择路径；逐单元重复上述操作。',
        ]),
    }
    for suffix, (question, facts) in edits.items():
        c = by['rag-biz-' + suffix]
        if question: c['question'] = question
        c['required_facts'] = facts
        c['answer_checks'] = [{'id': f'F{i}', 'role': 'core', 'text': t} for i,t in enumerate(facts,1)]
        c['annotation_provenance'] = {'source_version':'v4.2', 'review_date':'2026-11-10',
            'review_basis':'用户明确要求按已讨论方案修改019、033、040、049并重新测试。原reviewed状态保留；007/024不变。',
            'note':'仅修订题面和Gold，不修改检索、生成、判分算法；不由模型运行结果反推标准答案。'}
    by['rag-biz-033']['forbidden_claims'] = [
        '不核对方法学就把所有显示“＞”的项目判为竞争法、所有显示“＜”的项目判为夹心法。',
        '仅凭“＞/＜”就确定磁珠未混匀或底物漏液是唯一原因。',
        '仅因部分项目在控就永久排除仪器或质控品问题。',
    ]
    c = by['rag-biz-049']
    c['expected_behavior']='answer'
    c['missing_fields']=[]
    c['constraints']['object']='CL-6000i SDM 非运动参数备份'
    c['gold_evidence']=[e for e in c['gold_evidence'] if e['page']==35]
    c['forbidden_claims']=[
        '声称所有CL-6000i SDM版本都能一键保存全部单元参数。',
        '颠倒SDM1.0串口与SDM3.0网口的对应关系。',
        '将串口保存一个单元的参数说成已经备份整个模块。',
        '认为不提供现场SDM版本就完全不能说明两种备份方法。',
    ]
    doc['_阅读说明'].update({
        '用途':'Gold v4.3：按2026-11-10用户要求修订019/033/040/049；保留v4.2及更早版本。',
        '状态提示':'reviewed 48；007 disputed、024 pending，不计入正式评测。',
        '审核提示':'本轮仅修订四题，不切换应用默认测试集。',
        '本轮变更':'019拆分四个顺序要点且保留建议次数；033核对条件后分步排查；040只问部件方向；049改为备份方法题并允许分支回答。',
    })
    target=F/'rag_gold_v4_3_review.json'
    assert not target.exists(), 'Do not overwrite an existing revision'
    write(target,doc)
    (F/'rag_gold_v4_3.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in doc['cases']),encoding='utf-8')
    new=load_gold_dataset(target,reviewed_only=False)
    assert [c.raw for c in new.cases]==[c.raw for c in load_gold_dataset(F/'rag_gold_v4_3.jsonl',reviewed_only=False).cases]
    changes=[{'id':a['id'],'before':a,'after':b} for a,b in zip(old['cases'],doc['cases']) if a!=b]
    assert {r['id'] for r in changes}=={'rag-biz-'+s for s in edits}
    assert all(a['review_status']==b['review_status'] for a,b in zip(old['cases'],doc['cases']))
    reviewed=OUT/'formal_reviewed.jsonl'
    reviewed.write_text(''.join(json.dumps(c.raw,ensure_ascii=False)+'\n' for c in new.cases if c.review_status=='reviewed'),encoding='utf-8')
    assert len(load_gold_dataset(reviewed).cases)==48
    store=RagIndexStore(Path('E:/everythingfinder-db/rag')); bundle=store.active()
    with store.connect_read_only(bundle.metadata_path) as conn: chunks=load_chunk_rows(conn)
    audit=[]
    for c in new.cases:
        if c.case_id in {r['id'] for r in changes}: audit.extend(audit_case_sources(c,chunks,[]))
    write(OUT/'source_audit.json',audit)
    write(OUT/'changes.json',changes)
    write(OUT/'dataset_validation.json',{'cases':50,'unchanged_cases':46,'status_counts':dict(Counter(c.review_status for c in new.cases)),
        'json_jsonl_equal':True,'loader_full_diagnostic':'pass','loader_reviewed48':'pass',
        'behavior_counts_reviewed':dict(Counter(c.expected_behavior for c in new.cases if c.review_status=='reviewed')),
        'bundle':bundle.bundle_id,'historical_json_sha256':sha(F/'rag_gold_v4_2_review.json'),
        'new_json_sha256':sha(target),'new_jsonl_sha256':sha(F/'rag_gold_v4_3.jsonl')})
    lines=['# Gold v4.3 修订说明\n\n仅修改四题；其他46题、全部审核状态、历史文件和应用默认设置不变。007为disputed；024为pending。\n\n']
    for r in changes:
        c=r['after']; lines.extend([f"## {c['id']}\n\n{c['question']}\n\n标准答案要点：\n\n",*['- '+t+'\n' for t in c['required_facts']],'\n'])
    lines.append('本版本变化属于题面及评价标准修正，不能作为算法收益。原50题中48题正式计分；回归集pending仅作诊断。\n')
    (F/'rag_gold_v4_3.md').write_text(''.join(lines),encoding='utf-8')
    print(json.dumps(read(OUT/'dataset_validation.json'),ensure_ascii=False))
    print(json.dumps(audit,ensure_ascii=False)[:10000])

if __name__=='__main__': main()
