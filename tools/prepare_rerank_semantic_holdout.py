"""Freeze new source-derived questions; never edit approved suite or auto-review."""
import hashlib
import json
import re
import unicodedata
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from tools.validate_rerank_semantic import OUT, write, read_jsonl
from tools.rag_trace_audit import original_text


def compact(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))


def main():
    sources=json.loads((OUT/'holdout_source_candidates.json').read_text(encoding='utf8'))
    extras=json.loads((OUT/'holdout_extra_sources.json').read_text(encoding='utf8'))
    specs=[
        (sources[1]['candidates'][1], 'CL-2600i 底物本底偏高，光学组件已确认正常，单独测底物瓶的发光值超过6000，该怎么处理？',
         ['如果底物瓶中底物发光值>6000, 需要更换底物和使用“工程师维护底物管路”流程进行底物管路重新清洗和灌注','≥V01.00.06']),
        (sources[3]['candidates'][0], 'CL-2600i 自动抛杯组件老化测试，宏指令按什么顺序执行？',
         ['先输入指令 ACVT_Reset','结束后再输入指令 ACVT_Switch']),
        (sources[3]['candidates'][1], 'CL-2600i 预计停机超过两周，如何做整机排空？',
         ['整机排空时机：仪器停机时间>2周','工程维护软件→调试→整机排空→自动排空']),
        (sources[4]['candidates'][0], 'CL-2600i 意外断电后，模块恢复和重启操作软件都不能联机，电脑和仪器应该按什么顺序重启？',
         ['1）电脑关机','2）仪器关机','3）仪器重新上电','4）电脑开机','5）重新打开操作软件']),
        (sources[6]['candidates'][1], 'BS-2800M 搅拌杆清洗单元包含几个清洗池、节流阀和电磁阀？',
         ['搅拌杆清洗单元包括6个搅拌杆清洗池，2个节流阀，4个电磁阀']),
        (sources[9]['candidates'][1], '蠕动泵怎样让液体流动？更换泵管时能与其他泵随便混用吗？',
         ['通过对泵的弹性输送软管交替进行挤压和释放来泵送流体','蠕动泵及泵管要匹配不能混用']),
        (sources[10]['candidates'][0], '激光式与 CCD 式条码扫描器使用的发光源有什么区别？',
         ['激光式：以激光作为发光源','CCD式：以LED作为发光源']),
        (sources[11]['candidates'][0], '水质检测传感器是怎样把水质情况转成电阻率的？杂质增多时测得的电压如何变化？',
         ['对应的输出模拟电压的变化，由软件计算出电阻率','当水中杂质较多时，相当于水的电阻率较小，对应的电压较低']),
        (extras[0], '单向阀为什么不能让液体反向流动，工作时需要通电吗？',
         ['反之球体堵塞通道，液体和气体不能通过','单向阀工作时不需要通电']),
        (extras[1], 'CL-2600i 更换升版后的7号驱动板，需要什么操作软件版本？版本太低会有什么提示？',
         ['新板卡需使用≥V01.00.07操作软件','DMN烧写控制软件时会提示“驱动检测板升级失败！”','打开操作软件后会提示“电机驱动板7”异常']),
        (extras[2], '隔膜泵是怎样吸入和排出液体的？',
         ['当隔膜向下移动时，泵头内的容积增加，导致流体被吸入','当隔膜向上移动时，泵头内的容积减少，导致流体被排出']),
        (sources[2]['candidates'][1], 'BS-1000M 调光电采集位置，采集信号不满足要求时该调整哪里，然后做什么？',
         ['调整码盘传感器的左右位置，然后点击“重测”']),
    ]
    rows=[];audits=[]
    history=read_jsonl(OUT.parent/'rag_support_graph_20261113/A.jsonl')
    used={c['chunk_id'] for r in history for cs in r['candidate_stages'].values() for c in cs if 'chunk_id' in c}
    for i,(source,question,quotes) in enumerate(specs,1):
        s=source['source'];path=s['file_path'];slide=s['slide_number']
        # Resolve presentation relationship order, never assume slideN.xml is
        # the visible Nth slide (decks can be reordered).
        text=original_text(__import__('pathlib').Path(path),slide)
        evidence=[]
        for q in quotes:
            index_ok=q in source['text'];original_ok=compact(q) in compact(text)
            assert index_ok and original_ok,(i,q,index_ok,original_ok)
            evidence.append({'file':path,'page':slide,'position':f'第 {slide} 张幻灯片，正文',
                'quote':q,'relevance':2,'chunk_id':source['chunk_id']})
            audits.append({'id':f'rerank-new-{i:02}','file':path,'slide':slide,'quote':q,
                'original_text_verified':original_ok,'indexed_body_verified':index_ok,
                'layout_verified':'pending; native text verified, no automatic spatial proof',
                'source_sha256':hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest()})
        assert source['chunk_id'] not in used
        rows.append({'id':f'rerank-new-{i:02}','group_id':f'rerank-new-{i:02}', 'question':question,
            'corpus_snapshot':'g000033-71ec3c52f3','expected_behavior':'answer',
            'constraints':{'scope':'','stage':'','direction':'','object':''},
            'gold_evidence':evidence,'hard_negatives':[], 'required_facts':quotes,
            'forbidden_claims':['不得将本资料的型号、软件版本或条件扩展至其他场景'],
            'missing_fields':[],'review_status':'pending','source_context':source['text'],
            'independence':'new question and previously unused indexed chunk; source documents can overlap historical suite; pending diagnostic only'})
    target=OUT/'new_holdout_pending.jsonl'
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if target.exists():assert target.read_text(encoding='utf8')==payload
    else:target.write_text(payload,encoding='utf8')
    write(OUT/'new_holdout_source_audit.json',{'cases':len(rows),'quote_count':len(audits),'records':audits,
        'status':'pending; text verification is not human business approval','included_in_formal_metrics':False})
    print('Frozen pending questions:',len(rows),'verified quotes:',len(audits))

if __name__=='__main__':main()
