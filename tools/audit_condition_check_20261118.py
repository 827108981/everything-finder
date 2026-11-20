"""Source and selection audits, including negative and no-positive-Gold cases."""
from collections import Counter
from functools import lru_cache
import json
import re
from pathlib import Path
from tools.validate_condition_check_20261118 import OUT, BASE, ARMS, exp
from tools.validate_selection_factorial import write, sha
from tools.rag_trace_audit import original_text
from local_full_text_search.rag.content_identity import canonical_content


def main():
    base={r['id']:r for r in exp.read_rows(OUT/'baseline_native.jsonl')}
    sources={};changes={};hard_rejections={};rank_changes={};no_gold={}
    @lru_cache(maxsize=256)
    def original(path,page):
        try:
            p=Path(path)
            resolved=p
            method='read_original'
            if p.suffix.casefold() in {'.ppt','.doc','.xls'}:
                from local_full_text_search.parsers.legacy_office_parser import LegacyOfficeParser
                resolved=LegacyOfficeParser()._cached_conversion(p)
                if resolved is None:
                    raise ValueError('No source-fingerprint-matched legacy conversion cache; no conversion performed')
                method='source_stat_keyed_existing_conversion_cache'
            if resolved.suffix.casefold() in {'.xlsx','.xlsm'}:
                from local_full_text_search.parsers.xlsx_parser import XlsxParser
                from local_full_text_search.core.task_manager import CancelToken
                text='\n'.join(b.raw_text for b in XlsxParser().parse(resolved,CancelToken()))
            else:
                text=original_text(resolved,page)
            return dict(text=text,sha256=sha(p),error='',verification_method=method,resolved_path=str(resolved))
        except Exception as e:
            return dict(text='',sha256=None,error=str(e))
    @lru_cache(maxsize=256)
    def original_canonical(path,page):
        # Provenance-only verification: line-sized normalization avoids the
        # original offset builder's quadratic scans across a whole workbook.
        return ''.join(canonical_content(line) for line in original(path,page)['text'].splitlines())
    for arm in ARMS:
        rows=exp.read_rows(OUT/(arm+'_native.jsonl'))
        changes[arm]=[];hard_rejections[arm]=[];rank_changes[arm]=[];no_gold[arm]=[]
        for r in rows:
            old=base[r['id']]
            before={e['chunk_id']:e for e in old['entries']};after={e['chunk_id']:e for e in r['entries']}
            old_candidates={c['chunk_id']:c for c in old['candidate_stages']['constraints']}
            if not r['scores']['context']['gold_evidence_total_declared'] and bool(r['entries'])!=bool(old['entries']):
                no_gold[arm].append(dict(id=r['id'],question=r['question'],before_reason=old['decision']['reason'],after_reason=r['decision']['reason'],before_entries=len(before),after_entries=len(after)))
            oc=old['scores']['context'];nc=r['scores']['context']
            if oc['first_gold_rank']!=nc['first_gold_rank']:
                rank_changes[arm].append(dict(id=r['id'],before=oc['first_gold_rank'],after=nc['first_gold_rank']))
            for cid in sorted(before.keys()^after.keys()):
                c=next(c for c in r['candidate_stages']['constraints'] if c['chunk_id']==cid)
                source=c['file_path'];page=c.get('slide_number') or c.get('page_number')
                key=source+'#'+str(page)
                src=original(source,page)
                if key not in sources:sources[key]=dict(file=source,page=page,**src)
                text=after.get(cid,before.get(cid))['included_text']
                norm=canonical_content(text)
                checks=c.get('query_plan_diagnostics',{}).get('condition_relations',{}).get('checks',[])
                spans_valid=all(x['source_start'] is None or c['text'][x['source_start']:x['source_end']]==x['candidate_quote'] for x in checks)
                # A Word table serializes cells with |; verify each cell's text,
                # without claiming that flattened text proves visual ownership.
                cells=[canonical_content(p.strip()) for line in text.splitlines() for p in re.split(r' \| |\t',line) if p.strip()]
                original_cells=all(cell in original_canonical(source,page) for cell in cells)
                changes[arm].append(dict(id=r['id'],question=r['question'],change='added' if cid in after else 'removed',
                    chunk_id=cid,source_key=key,filename=c['filename'],page=page,location=c['location'],included_text=text,
                    original_text_verified=bool(norm and norm in original_canonical(source,page)),index_text_verified=norm in canonical_content(c['text']),
                    original_cells_verified=original_cells,original_method=src.get('verification_method','unavailable'),
                    source_error=src['error'],condition_decision=c.get('constraint_decision'),checks=checks,proof_spans_valid=spans_valid,
                    semantic_applicability='pending_source_review'))
            for c in r['candidate_stages']['constraints']:
                cd=c['constraint_decision'];oldc=old_candidates[c['chunk_id']]['constraint_decision']
                if not cd['accepted'] and oldc['accepted']:
                    hard_rejections[arm].append(dict(id=r['id'],question=r['question'],chunk_id=c['chunk_id'],file=c['file_path'],location=c['location'],
                        text=c['text'],decision=cd,proof=c.get('query_plan_diagnostics',{}).get('condition_relations')))
    write(OUT/'selection_audit.json',dict(changes=changes,new_hard_rejections=hard_rejections,rank_changes=rank_changes,no_gold_behavior_changes=no_gold,
        summary={a:dict(changes=len(v),added=sum(x['change']=='added' for x in v),removed=sum(x['change']=='removed' for x in v),
            quote_verified=sum(x['original_text_verified'] for x in v),source_errors=sum(bool(x['source_error']) for x in v),
            new_hard_rejections=len(hard_rejections[a]),no_gold_changes=no_gold[a]) for a,v in changes.items()}))
    write(OUT/'original_pages.json',sources)
    # A compact reading file: all A additions plus every new hard rejection.
    lines=[]
    for x in changes['A']:
        if x['change']=='added':
            lines.extend([f"\n## {x['id']} {x['question']}", f"{x['filename']} {x['location']} 原文核验={x['original_text_verified']}",x['included_text']])
    for x in hard_rejections['AB']:
        lines.extend([f"\n## 新硬排除 {x['id']} {x['question']}",x['file']+' '+x['location'],x['text'],json.dumps(x['proof'],ensure_ascii=False)])
    (OUT/'source_review_reading.md').write_text('\n\n'.join(lines),encoding='utf8')
    print(json.dumps({a:dict(changed=len(v),added=sum(x['change']=='added' for x in v),new_rejections=len(hard_rejections[a]),no_gold_changes=no_gold[a]) for a,v in changes.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
