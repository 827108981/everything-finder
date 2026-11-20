"""Identify changed generation inputs after the final generic boundary fixes."""
from dataclasses import replace
import json
from tools.rag_eval_common import ROOT,read,lines,write,sha
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.models import PlanSignal

OUT=ROOT/'reports/rag_adapter_robustness_20261111'

def main():
    before=read(OUT/'plans_before_method_fix.json')
    old={r['id']:r for r in lines(OUT/'native_v2.jsonl')}
    current=lines(OUT/'native_release.jsonl')
    rows=[]
    for r in current:
        p=QueryInterpreter(enhanced=True,domain_config='chemiluminescence').interpret(r['question'])
        pold=replace(p,strategies=tuple(PlanSignal(**s) for s in before[r['id']]['strategies']))
        summary_equal=AnswerService._plan_summary(p)==AnswerService._plan_summary(pold)
        context_equal=r['context']==old[r['id']]['context']
        rows.append({'id':r['id'],'context_equal':context_equal,'plan_summary_equal':summary_equal,
                     'generation_inputs_equal':context_equal and summary_equal})
    write(OUT/'release_input_delta.json',{'count':len(rows),'cases':rows,
        'changed_ids':[r['id'] for r in rows if not r['generation_inputs_equal']],
        'native_manifest_hash':sha(OUT/'native_release_manifest.json'),
        'reason':'Only unchanged actual generation requests may reuse the prior answer; changed requests require a new run'})
    print(len(rows),[r['id'] for r in rows if not r['generation_inputs_equal']])

if __name__=='__main__':main()
