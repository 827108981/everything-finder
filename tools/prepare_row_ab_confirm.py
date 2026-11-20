"""Freeze source-first diagnostic cases before any QA execution."""
import json,hashlib
from datetime import datetime
from pathlib import Path
from pptx import Presentation
from tools.validate_row_integration import ROOT,sha
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows
from local_full_text_search.rag.content_identity import canonical_content
OUT=ROOT/'reports/rag_row_ab_confirm_20261105'
FIX=ROOT/'tests/rag/fixtures/rag_row_holdout_v2_review.json'
def main():
 OUT.mkdir(exist_ok=False)
 units=json.loads((ROOT/'reports/rag_row_integration_20261105/row_index.json').read_text(encoding='utf8'))
 store=RagIndexStore(Path('E:/everythingfinder-db/rag'));b=store.active();conn=store.connect_read_only(b.metadata_path);chunks=load_chunk_rows(conn);conn.close()
 specs=[
 ([178],'免疫仪样本条码支持哪些条形码码制？',['支持Codabar、I 2 of 5、code128、code39、UPC/EAN、Code93。']),
 ([445],'CL-8000i光耦转接板的绿色状态灯常亮和熄灭，分别对应光耦什么状态？',['常亮表示对应光耦未遮挡，熄灭表示被遮挡。']),
 ([1372],'BS-2800M和BS-5000的反应体积范围分别是多少？',['BS-2800M为80–280μL。','BS-5000为70–240μL。']),
 ([1487],'BS-5000透射模块的卤钨灯标称寿命是多少小时？',['标称寿命为2400小时。']),
 ([2055,2056],'BS-360E和BS-430的反应盘分别用什么方式加热？',['BS-360E采用空气浴，BS-430采用固体直热。']),
 ([2063],'生化仪去离子水加热保护开关在多少温度关断，降到多少温度重新开启？',['60℃关断，低于30℃开启。']),
 ([2067],'BS-800反应盘温度传感器在37℃时正常阻值约多少，是正还是负温度系数？',['37℃时约1115Ω，正温度系数。']),
 ([2071],'生化仪压缩制冷模块红灯连续闪13次、间隔4秒后重复，是什么报警？什么条件会触发？',['是防冻报警。','内部蒸发器温度低于−20℃持续5分钟触发。']),
 ([1371],'BS-5000有多少个测量波长，覆盖什么范围？',['16个测量波长，范围340–850nm。']),
 ([445],'CL-8000i样本针水平零位传感器接在光耦转接板哪个接口？能换到其他接口吗？',['接在J1，位置一一对应，不能随意换到其他接口。'])]
 cases=[];audit=[];sources={}
 for n,(ids,q,facts) in enumerate(specs,1):
  ev=[]
  for i in ids:
   u=units[i];p=Presentation(u['file']);slide=p.slides[u['page']-1];tables=[s.table for s in slide.shapes if s.has_table]
   cells=[c.text.strip() for c in tables[u['table']-1].rows[u['row']-1].cells]
   assert cells==[x or '' for x in u['source_values']],(i,cells,u['source_values'])
   quote='\n'.join(u['source_values'])
   if n==10:quote='CL-8000i样本针水平零位传感器的插头在光耦转接板上的J1位置，不能随意插到其他位置'
   raw='\n'.join(s.text for s in slide.shapes if s.has_text_frame)+'\n'+'\n'.join(c.text for t in tables for r in t.rows for c in r.cells)
   assert canonical_content(quote) in canonical_content(raw)
   pagebody='\n'.join(x['text'] for x in chunks if x['file_path']==u['file'] and (x.get('slide_number') or x.get('page_number'))==u['page'])
   assert canonical_content(quote) in canonical_content(pagebody),(n,'index quote absent')
   ev.append({'file':u['file'],'page':u['page'],'position':f"原生表{u['table']}第{u['row']}行" if n!=10 else '传感器接线注意事项','quote':quote,'relevance':2,'why_sufficient':'原文对象、列名和数值直接对应问题，答案只要求所问事实。'})
   audit.append({'id':f'rag-row-holdout2-{n:03}','file':u['file'],'page':u['page'],'quote':quote,'original_verified':True,'index_verified':True,'original_cells':cells,'headers':[b['header'] for b in u['bindings']]})
   sources[u['file']]=sha(u['file'])
  if n==7:
   u=units[2067];s=Presentation(u['file']).slides[25];quote='BS-800';assert any(quote in sh.text for sh in s.shapes if sh.has_text_frame)
   ev.append({'file':u['file'],'page':26,'position':'反应盘加热机型标签','quote':quote,'relevance':2,'why_sufficient':'相邻连续反应盘加热页面标示BS-800；阻值表在下一张。'})
  cases.append({'id':f'rag-row-holdout2-{n:03}','group_id':f'row_holdout2_topic_{n:03}','question':q,'corpus_snapshot':b.bundle_id,'expected_behavior':'answer','constraints':{'scope':'','stage':'','direction':'','object':''},'missing_fields':[],'gold_evidence':ev,'hard_negatives':[],'required_facts':facts,'forbidden_claims':[],'answer_checks':[{'id':f'F{k}','role':'core','text':f} for k,f in enumerate(facts,1)],'review_status':'pending'})
 v={'_说明':'方案冻结后、首次问答前从原资料制题；全部pending。问题未参与调试，资料及行库已被使用；不是业务人员独立盲标，且主要覆盖表格场景。','cases':cases}
 FIX.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');FIX.with_name('rag_row_holdout_v2.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf8')
 (OUT/'source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
 paths=list((ROOT/'local_full_text_search/rag').rglob('*.py'))+[ROOT/'tools/validate_material_repairs.py',ROOT/'reports/rag_row_integration_20261105/row_index.json',FIX]
 from local_full_text_search.config.constants import SETTINGS_PATH
 paths+=[SETTINGS_PATH]
 (OUT/'freeze.json').write_text(json.dumps({'frozen_at':str(datetime.now()),'bundle_id':b.bundle_id,'hashes':{str(p):sha(p) for p in paths},'source_hashes':sources,'gold_status':'all pending','policy':'Frozen H8: BM25top10+native CE; at most one hard-match row in unused slot; preserve original;8sources/12000chars. No tuning after freeze. All48reviewed+10regression+10oldholdout+10newholdout, each arm independently fresh full AnswerService.answer including configured semantic clarification/review. Judge actual context v3.1. No latency analysis.'},ensure_ascii=False,indent=2),encoding='utf8')
 print('frozen',len(cases),b.bundle_id)
if __name__=='__main__':main()
