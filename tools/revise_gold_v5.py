"""User-approved business corrections; never derive labels from model answers."""
from copy import deepcopy
from collections import Counter
import json, re, sqlite3
from pathlib import Path
from pptx import Presentation
import fitz
from tools.rag_eval_common import ROOT, read, write, sha
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset

F=ROOT/'tests/rag/fixtures'
OUT=ROOT/'reports/rag_business_repair_20261112'
DB=Path('E:/everythingfinder-db/rag/rag.g000033-71ec3c52f3/metadata.db')
N=lambda s: re.sub(r'\s+','',s).casefold()

def main():
    sources=[F/'rag_gold_v4_3_review.json',F/'rag_acceptance_v1_review.json',F/'rag_row_holdout_v1_review.json',F/'rag_row_holdout_v2_review.json',ROOT/'reports/rag_evidence_answer_native_recheck_20261109/acceptance_review.json']
    names=['rag_gold_v5','rag_acceptance_v1_1','rag_row_holdout_v1_1','rag_row_holdout_v2_1','rag_independent_v2']
    docs=[deepcopy(read(p)) for p in sources]
    docs=[{'cases':d} if isinstance(d,list) else d for d in docs]
    old={c['id']:deepcopy(c) for d in docs for c in d['cases']}
    by={c['id']:c for d in docs for c in d['cases']}
    conn=sqlite3.connect('file:'+DB.as_posix()+'?mode=ro',uri=True); conn.row_factory=sqlite3.Row
    indexed=[]
    for r in conn.execute('select stable_chunk_id,text,source_json from rag_chunks'):
        indexed.append((r['stable_chunk_id'],r['text'],json.loads(r['source_json'])))
    audit=[]
    def evidence(path,page,start,end=None):
        hits=[(i,t,s) for i,t,s in indexed if N(s.get('path','').replace('\\','/'))==N(path.replace('\\','/')) and (s.get('slide_number') or s.get('page_number'))==page]
        if not hits: # SourceRef uses file_path in older bundles.
            hits=[(i,t,s) for i,t,s in indexed if path.split('/')[-1] in str(s) and (s.get('slide') or s.get('page') or s.get('slide_number') or s.get('page_number'))==page]
        assert hits,(path,page)
        i,t,s=next((i,t,s) for i,t,s in hits if start in t)
        a=t.index(start); b=t.index(end,a)+len(end) if end else len(t.rstrip())
        quote=t[a:b].strip()
        if path.endswith('.pptx'):
            slide=Presentation(path).slides[page-1]; parts=[]
            for shape in slide.shapes:
                if shape.has_text_frame: parts.append(shape.text)
                if shape.has_table: parts.extend('\n'.join(c.text for c in row.cells) for row in shape.table.rows)
            original='\n'.join(parts)
        else:
            with fitz.open(path) as pdf: original=pdf[page-1].get_text()
        assert N(quote) in N(original),(path,page,quote)
        audit.append({'file':path,'page':page,'quote':quote,'chunk_id':i,'original_match':True,'index_match':True,'source_sha256':sha(Path(path))})
        return {'file':path,'page':page,'position':'原文对应表格/正文；页码从1开始','quote':quote,'relevance':2,'why_sufficient':'原资料与已发布索引逐字核实；对象及字段/步骤对应本题。'}
    def update(id,q,facts,forbidden):
        c=by[id]; c.update(question=q,expected_behavior='answer',required_facts=facts,answer_checks=[{'id':f'F{i}','role':'core','text':t} for i,t in enumerate(facts,1)],forbidden_claims=forbidden,missing_fields=[],review_status='reviewed')
        c['annotation_provenance']={'review_date':'2026-11-12','review_basis':'用户本轮明确确认业务结论；按原资料和当前AI索引核实，不依据系统答案改标。','source_version':old[id].get('annotation_provenance',{}).get('source_version','previous fixture'),'model_alias_policy':'不推断BS-2800与BS-2800M等价，电压题使用手册准确型号。'}
        for key in ('source_unit_id','source_bindings'): c.pop(key,None)
        return c
    c=update('rag-biz-019','大部分发光项目定标失败，报 DEV，可能是什么原因？怎么解决？',[
        '可能原因包括仪器管路污染、分离液污染、底物液污染和仪器加样系统故障。',
        '先排查仪器相关因素，再排除故障和污染源。','排除故障和污染源后，对相关系统或液路执行灌注。','保留资料建议：灌注40次以上。','灌注后重新定标。'],['仅凭DEV就确定某一污染或加样故障已经发生。','未排除故障和污染源就把直接重新定标作为完整处理。'])
    p='E:/生化免疫临床知识/化学发光定标处理思路分析.pptx'
    c['gold_evidence']=[evidence(p,13,'大部分项目定标失败，报DEV','大部分项目定标失败，报DEV'),evidence(p,14,'可能原因：','40次以上')]
    c=update('rag-biz-050','CL-8000i 暗计数诊断失败，可能是什么原因？',['可能是UPS供电或者地线不符合要求。','可能是漏光。','可能是反应盘测光位有发光异物。','可能是光学异常。'],['仅凭暗计数失败断定光度计已损坏。','在没有进一步依据时指定必须更换的部件或物料编码。'])
    c['gold_evidence']=[evidence('E:/免疫资料/免疫场景化培训教材-光学V1.1.pdf',17,'失败可能原因：','光学异常')]
    c['constraints']['object']='CL-8000i 暗计数诊断'
    id=next(k for k in by if k.endswith('independent-07'))
    c=update(id,'A&T ISE 注射器密封圈的使用寿命是多少？物料编码是什么？',['注射器密封圈使用寿命为66000次测试，到寿命需更换。','注射器密封圈物料编码为040-000765-00。'],['把密封圈的66000次寿命和物料编码说成整套注射器的参数。'])
    p='E:/生化资料/生化场景化培训教材-电解质V1.1.pptx'
    c['gold_evidence']=[evidence(p,12,'ISE注射器的密封圈，物料编码是','040-000765-00。'),evidence(p,27,'注射器密封圈更换：','（到寿命需更换）')]; c['constraints']['object']='A&T ISE 注射器密封圈'
    c=update('rag-row-holdout-001','BS-2800M 的 P05 浓缩清洗剂泵，正常驱动电压是多少？',['P05浓缩清洗剂泵标称驱动电压为DC24V。','正常供电范围约22.8～25.2V，是24V标称值附近的正常范围。'],['认为24V标称值与22.8～25.2V正常范围互相矛盾。','未经依据将BS-2800和BS-2800M视为同一型号。'])
    p='E:/生化资料/BS-2600M&BS-2800M维修手册（中文）.pdf'
    c['gold_evidence']=[evidence(p,277,'P05','00096-00 \nDC24V'),evidence(p,205,'表 5.77 J7','0V）时P05 开')]; c['constraints']['object']='BS-2800M P05 浓缩清洗剂泵'
    c['source_scope']={'models':['BS-2600M','BS-2800M'],'basis':'维修手册封面及文件名；不登记BS-2800别名'}
    c=update('rag-row-holdout2-001','免疫仪样本条码支持哪些条形码码制？',['高速和中低速免疫资料共同列出Codabar、I 2 of 5、Code128、Code39、UPC/EAN、Code93。'],['把共同条码码制限定为仅CL-2600i支持。','把单一机型独有规则推广到全部免疫仪。','不同资料不一致时将支持码制直接取并集作为共同标准。'])
    c['gold_evidence']=[evidence('E:/免疫资料/'+folder+'/10-学习参考资料包/'+name,pg,'支持条形码码制','Code93') for folder,name,pg in [('高速免疫','高速免疫场景化培训教材-条码V1.1.pptx',17),('中低速免疫','中低速免疫场景化培训教材-条码V1.1.pptx',14)]]
    c['evidence_requirements']={'distinct_sources_min':2,'platform_groups':['高速免疫','中低速免疫'],'consistency_required':True}; c['constraints']['object']='免疫仪样本条码码制'
    docs[0]['cases']=[c for c in docs[0]['cases'] if c['id']!='rag-biz-040']
    manifest={'version':'business-gold-v5-20261112','bundle_id':'g000033-71ec3c52f3','deleted_ids':['rag-biz-040'],'comparison_policy':'仅相同新版题目集合可作算法比较；旧50题与新49题不可直接比较总分。','datasets':[]}
    for source,name,d in zip(sources,names,docs):
        # Keep field definitions; replace stale version/count instructions.
        d={k:v for k,v in d.items() if k in ('cases','_字段说明','_取值说明')}
        d['_说明']={'version':manifest['version'],'previous_file':str(source),'cases':len(d['cases']),'review_policy':'本轮五题业务结论获用户明确确认，其余状态保持不变；040从活动集合删除。'}
        write(F/(name+'_review.json'),d)
        target=F/(name+'.jsonl'); target.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in d['cases']),encoding='utf8')
        a=load_gold_dataset(target,reviewed_only=False); b=load_gold_dataset(F/(name+'_review.json'),reviewed_only=False)
        assert [c.raw for c in a.cases]==[c.raw for c in b.cases]
        manifest['datasets'].append({'path':str(target.relative_to(ROOT)).replace('\\','/'),'count':len(a.cases),'status_counts':dict(Counter(c.review_status for c in a.cases)),'sha256':sha(target)})
    current=[c for d in docs for c in d['cases']]
    (F/'rag_gold_v5_reviewed.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in docs[0]['cases'] if c['review_status']=='reviewed'),encoding='utf8')
    assert len(current)==91 and len({c['id'] for c in current})==91
    write(F/'rag_active_suite_v5.json',manifest)
    write(OUT/'source_audit.json',audit)
    write(OUT/'gold_changes.json',[{'id':c['id'],'before':old[c['id']],'after':c} for c in current if c!=old[c['id']]]+[{'id':'rag-biz-040','before':old['rag-biz-040'],'after':None}])
    write(OUT/'dataset_validation.json',{'total':91,'main':49,'json_jsonl_equal':True,'loader':'pass','status_counts':dict(Counter(c['review_status'] for c in current)),'historical_hashes':{str(p):sha(p) for p in sources}})
    print('Created v5: main49 + regression42 =91; source quotations',len(audit))

if __name__=='__main__': main()
