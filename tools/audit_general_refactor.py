"""Audit immutable inputs, runtime boundaries, and disabled compatibility."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/rag_general_refactor_20261113'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    protected=[{'path':p,'before':h,'after':sha(Path(p)) if Path(p).is_file() else None} for p,h in freeze['protected_hashes'].items()]
    literals=[];imports=[]
    for path in (ROOT/'local_full_text_search/rag').rglob('*.py'):
        if 'evaluation' in path.parts: continue
        tree=ast.parse(path.read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                if re.search(r'rag-biz-\d+|row-holdout\d*[-_]|independent-\d+|CSDB|\bDEV\b|\bDUP\b|\bLVC\b|CL-2600i|CL-8000i|BS-2800M|\bP05\b|66000|040-000765|22\.8|25\.2',node.value):
                    literals.append({'path':str(path.relative_to(ROOT)),'line':node.lineno,'value':node.value[:180]})
            if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('local_full_text_search.rag.evaluation'):
                imports.append({'path':str(path),'line':node.lineno,'module':node.module})
    # Same frozen questions, no model or retrieval involved: compare disabled parsers.
    script=r'''
from dataclasses import asdict
from pathlib import Path
import json,sys
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.models import EvidenceChunk,RagChunk,SourceRef
root=Path(sys.argv[1]);out=Path(sys.argv[2]);manifest=json.loads((root/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
cases=[json.loads(l) for d in manifest['datasets'] for l in (root/d['path']).read_text(encoding='utf8').splitlines() if l.strip()]
result={c['id']:asdict(QueryInterpreter(enhanced=False).interpret(c['question'])) for c in cases}
out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True),encoding='utf8')
'''
    for name,cwd in [('A',Path(freeze['snapshot'])),('B',ROOT)]:
        subprocess.run([sys.executable,'-c',script,str(ROOT),str(OUT/('disabled_'+name+'_plans.json'))],cwd=cwd,check=True)
    a=json.loads((OUT/'disabled_A_plans.json').read_text(encoding='utf8'));b=json.loads((OUT/'disabled_B_plans.json').read_text(encoding='utf8'))
    result={'protected_count':len(protected),'protected_drift':[p for p in protected if p['before']!=p['after']],
            'runtime_case_literals':literals,'runtime_evaluation_imports':imports,
            'disabled_query_count':len(a),'disabled_plan_differences':[k for k in a if a[k]!=b[k]],
            'limitations':['Static scan checks known suspect values and imports, not proof that all possible semantic special cases are absent.',
                          'Disabled plan equality alone is not a complete end-to-end retrieval equivalence claim.']}
    from local_full_text_search.config.defaults import AppSettings
    from local_full_text_search.config.constants import SETTINGS_PATH
    from local_full_text_search.rag.index_store import RagIndexStore
    defaults=AppSettings();saved=json.loads(SETTINGS_PATH.read_text(encoding='utf8'))
    result['defaults']={k:getattr(defaults,k) for k in ('rag_query_plan_enabled','rag_structured_answer_enabled')}
    result['persisted_flags']={k:saved.get(k,False) for k in result['defaults']}
    index=json.loads((OUT/'index_freeze.json').read_text(encoding='utf8'))
    result['bundle_unchanged']=RagIndexStore('E:/everythingfinder-db/rag').active().bundle_id==index['bundle_id'] and sha(Path(index['metadata_path']))==index['metadata_sha256']
    (OUT/'architecture_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
