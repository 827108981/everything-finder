"""Continuous resumable batch runner; never publishes an index or calls an API."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback
from tools.validate_vector_packaging_20261116 import OUT,ROOT,read,write,inputs,sha,lines,deltas

def now():return datetime.now(timezone.utc).isoformat()

def run(stage,extra=None):
    name=stage+('_'+'_'.join(extra or []) if extra else '')
    path=OUT/(name.replace('/','_')+'.log')
    write(OUT/'execution_status.json',dict(status='in_progress',current=name,updated=now()))
    print(now(),'START',name,flush=True)
    with path.open('a',encoding='utf8') as f:
        result=subprocess.run([sys.executable,'-X','utf8','-m','tools.validate_vector_packaging_20261116',stage,*(extra or [])],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; see {path}')
    print(now(),'FINISHED',name,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    lock=OUT/'run.lock'
    if lock.exists():
        # A stale lock requires explicit resume, preventing accidental duplicate
        # encoders or SQLite/Chroma writers.
        import ctypes
        oldpid=int(lock.read_text().strip())
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x1000,False,oldpid)
        active=False
        if handle:
            code=ctypes.c_ulong()
            if kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code)):active=code.value==259
            kernel.CloseHandle(ctypes.c_void_p(handle))
        if active:raise RuntimeError(f'Runner {oldpid} is still active; refusing duplicate writers')
        if not a.resume:raise RuntimeError('Stale runner lock exists; use --resume after inspecting status')
    lock.write_text(str(__import__('os').getpid()),encoding='utf8')
    try:
        inputs()
        manifest=OUT/'implementation_manifest.json'
        files=[*Path(ROOT/'local_full_text_search/rag').rglob('*.py'),Path(__file__),
               *[ROOT/'tools'/name for name in ('validate_vector_packaging_20261116.py',
                 'packaging_encoding_runtime.py','finalize_vector_packaging_20261116.py',
                 'report_vector_packaging_20261116.py','check_packaging_report_ui.py',
                 'check_vector_packaging_report.cjs')]]
        hashes={str(p):sha(p) for p in files}
        if manifest.exists():assert read('implementation_manifest.json')['hashes']==hashes,'Implementation drift; do not silently resume with different runtime'
        else:write(manifest,dict(hashes=hashes,started=now(),api_calls=0))
        if not (OUT/'encoding_complete.json').exists():run('encode')
        for arm in [f'W{i}' for i in range(6)]:
            if not (OUT/f'{arm}_exact.jsonl').exists():run('exact',['--arms',arm])
        if not (OUT/'ann_audit.json').exists():run('ann')
        if not (OUT/'selection.json').exists():run('choose')
        selection=read('selection.json')
        if selection['dual_eligible']:
            # Never silently skip a triggered factor. A complete dual-view
            # implementation is required before the experiment can finish.
            run('dual')
        else:write(OUT/'dual_audit.json',dict(status='not_applicable',reason=selection['dual_reason']))
        if not (OUT/'factorial_protocol.json').exists():run('structure')
        # New structured rows may introduce additional exact encoding strings.
        run('encode')
        for arm in ('F10','F11','F10_count_matched'):
            if not (OUT/f'{arm}_exact.jsonl').exists():run('exact',['--arms',arm])
        if not (OUT/'F11_ann.jsonl').exists():run('ann',['--arms','F10','F11','F10_count_matched'])
        holdoutarms=list(dict.fromkeys(['W0',selection['packaging_candidate'],'F10','F11']))
        if not all((OUT/f'{arm}_exact_holdout.jsonl').exists() for arm in holdoutarms):run('exact',['--holdout','--arms',*holdoutarms])
        if not all((OUT/f'{arm}_ann_holdout.jsonl').exists() for arm in holdoutarms):run('ann',['--holdout','--arms',*holdoutarms])
        if not (OUT/'native_audit.json').exists():run('downstream',['--arms',*holdoutarms])
        subprocess.run([sys.executable,'-X','utf8','-m','tools.finalize_vector_packaging_20261116'],cwd=ROOT,check=True)
        print(now(),'ALL BATCHES FINISHED',flush=True)
    except BaseException as e:
        write(OUT/'execution_status.json',dict(status='failed',current=str(e),traceback=traceback.format_exc(),updated=now(),api_calls=0))
        # Keep native HTML truthful even when execution cannot complete.
        subprocess.run([sys.executable,'-X','utf8','-m','tools.report_vector_packaging_20261116'],cwd=ROOT)
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
