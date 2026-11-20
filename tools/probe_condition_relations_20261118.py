"""Freeze new source-page relationship probes before judging; pending only."""
import argparse
import json
from pathlib import Path
import re
from dataclasses import asdict
from tools.validate_condition_check_20261118 import OUT, BASE, exp
from tools.validate_selection_factorial import write, sha, no_network
from tools.rag_trace_audit import original_text
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.condition_relations import judge_relation
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.models import QueryConstraint, ConstraintKind as K

FIXTURE=OUT/'independent_relations_pending.json'


def freeze():
    if FIXTURE.exists():raise ValueError('Already frozen')
    store,bundle,_=exp.inputs()
    used={c['chunk_id'] for r in exp.read_rows(BASE) for c in r['candidate_stages']['rerank']}
    probes=[];seen=set()
    pattern=re.compile(r'(?:^|[。；;\n])\s*([A-Za-z0-9\u3400-\u9fff]{2,10})(偏低|偏高)[。；;：:，,]')
    with store.connect_read_only(bundle.metadata_path) as conn:
        rows=conn.execute("SELECT stable_chunk_id,text FROM rag_chunks WHERE text LIKE '%偏低%' OR text LIKE '%偏高%' ORDER BY stable_chunk_id")
        for row in rows:
            if row[0] in used:continue
            for m in pattern.finditer(row[1]):
                subject=m.group(1)
                if any(w in subject for w in ['如果','可能','原因','导致','认为','情况','问题','建议','结果使']):continue
                ch=store.load_chunk(conn,row[0]);src=ch.source
                page=src.slide_number or src.page_number
                key=(src.file_path,page)
                if key in seen or Path(src.file_path).suffix.casefold() not in {'.pptx','.pdf'}:continue
                quote=m.group().strip('。；;\n ')
                try:orig=original_text(Path(src.file_path),page)
                except Exception:continue
                if canonical_content(quote) not in canonical_content(orig):continue
                # Validate attribution inside this explicit source statement only.
                # This does not label an entire multi-scene page as inapplicable.
                direction='low' if m.group(2)=='偏低' else 'high'
                other='偏高' if direction=='low' else '偏低'
                unrelated='供电电压' if '温度' in subject else '环境温度'
                for task,q,value,status in (
                    ('same_value',subject+m.group(2)+'应该怎么检查？',direction,'match'),
                    ('opposite_same_attribute',subject+other+'应该怎么检查？','high' if direction=='low' else 'low','conflict'),
                    ('different_attribute',unrelated+other+'应该怎么检查？','high' if direction=='low' else 'low','unknown')):
                    probes.append(dict(id=f'condition-source-probe-{len(probes)+1:02}',question=q,
                        constraint=asdict(QueryConstraint(K.DIRECTION,value,source_text=m.group(2) if task=='same_value' else other)),
                        candidate_quote=quote,chunk_id=row[0],file=src.file_path,page=page,
                        source_sha256=sha(src.file_path),expected_status=status,category=task,review_status='pending',
                        original_quote_verified=True,index_quote_verified=quote in ch.text,
                        scope='Source statement applicability only; new page outside all 111 rerank entries; same corpus, not an unseen-document split'))
                seen.add(key);break
            if len(seen)>=6:break
    assert len(probes)>=3, len(probes)
    write(OUT/'independent_relations_coverage.json',dict(planned_source_pages=6,eligible_source_pages=len(seen),
        cases=len(probes),status='passed' if len(seen)>=6 else 'coverage_incomplete',
        limitation='Strict source and attribution criteria; do not pad with unsupported text or claim broad unseen-source acceptance'))
    write(FIXTURE,probes)
    print(json.dumps([dict(id=x['id'],question=x['question'],quote=x['candidate_quote'],file=x['file'],page=x['page']) for x in probes],ensure_ascii=False,indent=2))


def run():
    frozen=sha(FIXTURE);records=[]
    with no_network() as attempts:
        for f in json.loads(FIXTURE.read_text(encoding='utf8')):
            c=f['constraint'];constraint=QueryConstraint(K(c['kind']),c['value'],source_text=c['source_text'])
            old=ConstraintMatcher()._judge(constraint,f['candidate_quote'],f['candidate_quote'])
            new,proof=judge_relation(constraint,f['candidate_quote'],f['question'])
            records.append(dict(id=f['id'],expected=f['expected_status'],old_status=old.status.value,
                new_status=new.status.value,old_pass=old.status.value==f['expected_status'],new_pass=new.status.value==f['expected_status'],proof=proof))
    assert sha(FIXTURE)==frozen and not attempts
    write(OUT/'independent_relations_results.json',dict(fixture_sha256=frozen,cases=len(records),records=records,
        old_pass=sum(x['old_pass'] for x in records),new_pass=sum(x['new_pass'] for x in records),review_status='pending',
        api_calls=0,network_attempts=attempts))
    print('RELATION_PROBES',len(records),'old',sum(x['old_pass'] for x in records),'new',sum(x['new_pass'] for x in records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    freeze() if a.freeze else run()
