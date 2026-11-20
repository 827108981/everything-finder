"""Recover transport failures only, preserving original failed records."""
from pathlib import Path
import json,time,runpy,shutil
ROOT=Path('E:/code/everything-finder');P=ROOT/'reports/rag_row_ab_confirm_20261105'
def main():
 while not (P/'completion.json').exists():time.sleep(3)
 rs=[json.loads(l) for l in (P/'cases.jsonl').read_text(encoding='utf8').splitlines()]
 failed={(r['id'],r['arm']) for r in rs if r.get('error') and not r.get('answer') and any(t in r['error'].lower() for t in ['超时','timeout','connection','502','503','504'])}
 d=P/'transport_recovery';d.mkdir(exist_ok=True)
 (d/'selection.json').write_text(json.dumps({'failed_execution_only':sorted(failed),'runtime_timeout_seconds':180,'saved_settings_changed':False,'quality_results_retried':False},ensure_ascii=False,indent=2),encoding='utf8')
 if not failed:return
 shutil.copyfile(P/'freeze.json',d/'freeze.json')
 s=(ROOT/'tools/confirm_row_ab.py').read_text(encoding='utf8')
 s=s.replace("OUT=ROOT/'reports/rag_row_ab_confirm_20261105'","OUT=ROOT/'reports/rag_row_ab_confirm_20261105/transport_recovery'")
 s=s.replace("settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8')))","settings=replace(AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8'))),rag_api_timeout_seconds=180)")
 s=s.replace("if (c.case_id,arm,1) not in done]","if (c.case_id,arm,1) not in done and (c.case_id,arm) in "+repr(failed)+"]")
 target=ROOT/'tools/confirm_row_ab_transport_recovery.py';target.write_text(s,encoding='utf8')
 runpy.run_path(str(target),run_name='__main__')
if __name__=='__main__':main()
