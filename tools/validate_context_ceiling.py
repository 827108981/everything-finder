"""Diagnostic paired replay. Gold locates source pages, never enters generation prompts."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile
import xml.etree.ElementTree as ET

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rag_context_ceiling_20261104'
FIX=ROOT/'tests/rag/fixtures'
AB=ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl'
FULL=ROOT/'reports/rag_full_revalidation_20261103/cases.jsonl'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p): return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines()]
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def cases():
    return {c.case_id:c for name in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json')
            for c in load_gold_dataset(FIX/name,reviewed_only=False).cases}

def source_text(path,page):
    if path.suffix.lower()=='.pdf':
        import fitz
        with fitz.open(path) as d:return d[page-1].get_text()
    if path.suffix.lower()=='.pptx':
        # Resolve presentation order rather than assuming slideN.xml is slide N.
        with zipfile.ZipFile(path) as z:
            ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
                'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
                'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
            rel={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('ppt/_rels/presentation.xml.rels'))}
            slide=ET.fromstring(z.read('ppt/presentation.xml')).findall('.//p:sldId',ns)[page-1]
            target=rel[slide.attrib['{'+ns['r']+'}id']]
            name=target.lstrip('/') if target.startswith('/') else 'ppt/'+target
            xml=ET.fromstring(z.read(name))
            return '\n'.join(''.join(t.text or '' for t in p.findall('.//a:t',ns)) for p in xml.findall('.//a:p',ns))
    raise ValueError('Unsupported original source: '+str(path))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.jsonl').exists()
    cs=cases(); baseline={r['id']:r for r in lines(AB) if r['arm']=='baseline'}
    previous=lines(FULL)
    failed={r['id'] for r in previous+list(baseline.values()) if (r.get('gold_judgement')or{}).get('verdict')!='consistent'}
    # 005 has an explicitly approved annotation-only correction, not a production defect.
    failed.discard('rag-biz-005')
    failed={i for i in failed if cs[i].review_status!='disputed'}
    store=RagIndexStore(Path('E:/everythingfinder-db/rag')); bundle=store.active()
    rows=load_chunk_rows(store.connect_read_only(bundle.metadata_path)); byid={r['chunk_id']:r for r in rows}
    protocol={'started':str(datetime.now()),'bundle_id':bundle.bundle_id,'case_ids':sorted(failed),
      'selection':'Union of non-consistent cases in two pre-existing baselines; 005 approved label correction excluded; regression pending, diagnostic only.',
      'repeats':3,'arms':['current','source_completed'],'generation_prompt':'Unchanged production generate() and query plan summary. No Gold answers, rubrics, behavior labels or missing fields provided to generator.',
      'intervention':'Full verified Gold-source index pages first, then unchanged current entries within original 8 chunks / 12000 chars. Necessary preceding scenario page included for 019. May change order and remove distractors: diagnostic bound, not pure recall effect or production candidate.',
      'refusal':'No fabricated positive source; unchanged context in both arms; identity control.',
      'stop':'No quality-based retries, no prompt tuning; transport/JSON exceptions separately recorded. Same model judging has correlated-error limitation.',
      'hashes':{str(p):sha(p) for p in [SETTINGS_PATH,AB,FULL,FIX/'rag_gold_v4_1_review.json',FIX/'rag_gold_v4_1.jsonl',FIX/'rag_acceptance_v1_review.json',store.manifest_path]+list((ROOT/'local_full_text_search/rag').rglob('*.py'))}}
    audits=[]; prepared=[]; original_cache={}
    for cid in sorted(failed):
        c=cs[cid];b=baseline[cid]; selected=[]
        mappings=map_case_evidence(c,rows)
        for n,e in enumerate(c.gold_evidence,1):
            m=next(m for m in mappings if m.evidence_id==f'{cid}:positive:{n}')
            ids=m.chunk_ids
            assert ids,(cid,e)
            declared=[byid[i] for i in ids if Path(byid[i]['file_path']).name.casefold()==Path(str(e['file']).replace('\\','/')).name.casefold()]
            exact_path=[r for r in declared if r['file_path'].replace('\\','/').casefold()==str(e['file']).replace('\\','/').casefold()]
            if exact_path:declared=exact_path
            assert declared
            path=Path(declared[0]['file_path']);page=int(e['page']);cachekey=(str(path),page)
            if cachekey not in original_cache:original_cache[cachekey]=source_text(path,page)
            orig=original_cache[cachekey];indextext='\n'.join(r['text'] for r in sorted(declared,key=lambda r:r['ordinal']))
            q=canonical_content(e['quote'])
            audits.append({'id':cid,'evidence':n,'file':str(path),'page':page,'position':e.get('position'),
               'quote':e['quote'],'original_quote_exact_canonical':q in canonical_content(orig),
               'index_quote_exact_canonical':q in canonical_content(indextext),'original_page_text':orig,
               'index_page_text':indextext,'chunk_ids':[r['chunk_id'] for r in declared],
               'original_sha256':sha(path),'current_quote_present':q in canonical_content(b['context'])})
            selected.extend(declared)
        # Cross-page title supplies the real scope; no invented explanation prepended.
        if cid=='rag-biz-019':
            path=selected[0]['file_path']
            selected=[r for r in rows if r['file_path']==path and r['slide_number']==13]+selected
        # All original source page text must be audited before remote execution.
        unique={r['chunk_id']:r for r in selected}; priority=list(unique.values())
        entries=[{'chunk_id':r['chunk_id'],'filename':r['filename'],'location':r['location'],'included_text':r['text'],'truncated':False} for r in priority]
        for e in b['context_entries']:
            if e['chunk_id'] not in unique:entries.append(e)
        if not c.gold_evidence:
            context=b['context'];kept=b['context_entries'];removed=[]
        else:
            sections=[];kept=[];removed=[]
            for e in entries:
                sourceid=f'S{len(kept)+1}'
                text=f'[{sourceid}]\n来源（1 处）：\n- {e["filename"]} · {e["location"]}\n内容：{e["included_text"].strip()}'
                if len(kept)>=8 or len('\n\n'.join(sections+[text]))>12000:
                    assert e['chunk_id'] not in unique,('Gold page budget exceeded',cid)
                    removed.append(e['chunk_id']);continue
                sections.append(text);kept.append({**e,'source_id':sourceid})
            context='\n\n'.join(sections)
        prepared.append({'id':cid,'question':c.question,'expected_behavior':c.expected_behavior,
            'review_status':c.review_status,'cohort':'formal' if c.review_status=='reviewed' else 'regression_pending',
            'historical_verdicts':[(r['gold_judgement']['verdict']) for r in previous if r['id']==cid]+[b['gold_judgement']['verdict']],
            'current':{'context':b['context'],'context_entries':b['context_entries']},
            'source_completed':{'context':context,'context_entries':kept},'removed_original_ids':removed,
            'added_ids':[e['chunk_id'] for e in kept if e['chunk_id'] not in {x['chunk_id'] for x in b['context_entries']}],
            'same_context':context==b['context']})
    dump(OUT/'protocol.json',protocol);dump(OUT/'source_audit.json',audits);dump(OUT/'prepared.json',prepared)
    print('Prepared',len(prepared),'cases,',len(audits),'evidence checks')
    for a in audits:
        print(a['id'],a['page'],a['original_quote_exact_canonical'],a['index_quote_exact_canonical'],a['file'])

def run():
    protocol=read(OUT/'protocol.json');assert read(OUT/'audit_approval.json')['source_audit_complete']
    assert all(sha(p)==h for p,h in protocol['hashes'].items())
    s=AppSettings.from_dict(read(SETTINGS_PATH));key=WindowsCredentialSecretStore().retrieve(s.rag_api_secret_ref) or '';assert key
    cs=cases();prepared=read(OUT/'prepared.json')
    dump(OUT/'execution.json',{'started':str(datetime.now()),'model':s.rag_api_model,'workers':6})
    def task(p,arm,repeat):
        row={'id':p['id'],'arm':arm,'repeat':repeat,'cohort':p['cohort'],'error':None};t=time.monotonic()
        g=OpenAICompatibleGateway(max_transient_retries=2);c=cs[p['id']];data=p[arm]
        try:
            result=g.generate(GenerationRequest(question=c.question,context=data['context'],model=s.rag_api_model,
              api_base_url=s.rag_api_base_url,api_style=s.rag_api_style,api_key=key,timeout_seconds=s.rag_api_timeout_seconds,
              query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(c.question))),CancelToken())
            row['answer']={'mode':'full','text':result.text,'status':'answered','source_ids':[e['source_id'] for e in data['context_entries']]}
            row['citation_check']=asdict(CitationValidator().inspect(result.text,{e['source_id']:() for e in data['context_entries']}))
            row['gold_judgement']=GoldAnswerJudge(g,s,key).evaluate(c,row['answer'],CancelToken())
        except Exception as ex:row['error']={'type':type(ex).__name__,'message':str(ex)[:400].replace(key,'[REDACTED]')}
        row['elapsed_ms']=round((time.monotonic()-t)*1000);return row
    done=set()
    if (OUT/'results.jsonl').exists():done={(r['id'],r['arm'],r['repeat']) for r in lines(OUT/'results.jsonl')}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(task,p,arm,repeat) for repeat in range(1,4) for p in prepared for arm in ('current','source_completed') if (p['id'],arm,repeat) not in done]
        for f in as_completed(futures):
            r=f.result()
            with (OUT/'results.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
            print(r['id'],r['arm'],r['repeat'],(r.get('gold_judgement')or{}).get('verdict'),r['error'],flush=True)
    assert all(sha(p)==h for p,h in protocol['hashes'].items())
    results=lines(OUT/'results.jsonl');dump(OUT/'completion.json',{'finished':str(datetime.now()),'rows':len(results),'expected':len(prepared)*6,
      'hashes_unchanged':True,'errors':sum(bool(r['error']) for r in results),'no_production_or_gold_edits':True})

def retry_errors():
    """One exception-only recovery; retain successful answers if only judging failed."""
    s=AppSettings.from_dict(read(SETTINGS_PATH));key=WindowsCredentialSecretStore().retrieve(s.rag_api_secret_ref) or '';assert key
    cs=cases();prepared={p['id']:p for p in read(OUT/'prepared.json')}
    path=OUT/'execution_retries.jsonl'; retries=lines(path) if path.exists() else []
    tried={(r['id'],r['arm'],r['repeat']) for r in retries}
    for orig in lines(OUT/'results.jsonl'):
        if not orig['error'] or (orig['id'],orig['arm'],orig['repeat']) in tried:continue
        r=dict(orig);r['initial_error']=r['error'];r['error']=None
        g=OpenAICompatibleGateway(max_transient_retries=2);c=cs[r['id']];data=prepared[r['id']][r['arm']]
        r['retry_scope']='judge_only' if r.get('answer') else 'generation_and_judge'
        try:
            if not r.get('answer'):
                result=g.generate(GenerationRequest(question=c.question,context=data['context'],model=s.rag_api_model,
                  api_base_url=s.rag_api_base_url,api_style=s.rag_api_style,api_key=key,timeout_seconds=s.rag_api_timeout_seconds,
                  query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(c.question))),CancelToken())
                r['answer']={'mode':'full','text':result.text,'status':'answered','source_ids':[e['source_id'] for e in data['context_entries']]}
                r['citation_check']=asdict(CitationValidator().inspect(result.text,{e['source_id']:() for e in data['context_entries']}))
            r['gold_judgement']=GoldAnswerJudge(g,s,key).evaluate(c,r['answer'],CancelToken())
        except Exception as ex:r['error']={'type':type(ex).__name__,'message':str(ex)[:400].replace(key,'[REDACTED]')}
        with path.open('a',encoding='utf-8') as out:out.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
        print('exception recovery',r['id'],r['arm'],r['repeat'],(r.get('gold_judgement')or{}).get('verdict'),r['error'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','retry']);a=p.parse_args()
    {'prepare':prepare,'run':run,'retry':retry_errors}[a.action]()
