"""Question/evidence based rubric proposals; historical Gold remains immutable.

No generated answer or system result is read while making these proposals.
All proposed labels require human review and are not the application default.
"""
from __future__ import annotations
import copy
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.index_store import RagIndexStore

FIXTURES=Path('tests/rag/fixtures')


def check(cid, role, text):
    return {'id':cid,'role':role,'text':text}


def main():
    source=FIXTURES/'rag_gold_v3.jsonl'
    original=source.read_bytes()
    cases=[json.loads(l) for l in original.decode('utf-8').splitlines()]
    # Each proposal is based on the scope of the question and its existing quote.
    # Safety proposals and conditional case generalizations are explicitly disputed.
    reasons={
        1:'用户已说明上电、接线且灯灭；正常灯态是背景，标准模式、端口排查与更换前提仍是必答。',
        2:'只问D63/D64含义，上电后应亮起是可选背景。',
        3:'问参数、IP和序列号，并未询问拆装方法；断电和防静电不能因资料同页就自动成为本题必答。若答案涉及实际拆装，安全要求仍适用，需业务复核。',
        14:'唯一数值取决于盘标识，必须追问；追问时没有必要同时提供两个数值，数值分支改为背景。',
        21:'必须请求决定前次/本次异常的检查信息；LVC定义可辅助说明，但不是发出澄清的必备内容。',
        24:'询问发光值如何得到浓度或阴阳性，不是询问校准点数；转换关系必答，点数为背景。',
        27:'用户已给复测结果，仅问结果判定；复测前离心条件属于流程背景，不强制复述。判定适用性仍受原资料限制，不能推广到任意检测方法。',
        32:'问如何排查，应检查其他项目、历史/反应数据、试剂装载与混匀；不要求复述特定医院已发生的结局，也不能把案例当作当前诊断。',
        33:'问方向含义及先检查什么，须结合方法判断并检查磁珠混匀；历史停电维修及未进入软件是个案背景。',
        37:'问是否反复校准及核查方面；排除原因后再校准和核查列表为必答，不强制复述术语及记录要求。',
        38:'问先考虑什么，定标液复溶稳定性和保存条件为必答；每天校准并非保证、天间校准引入不确定因素是背景。',
        41:'问分别检查什么；水质和反应盘温度为必答，其他酶项目及其他项目正常等说明为背景。',
        46:'只问杯位数及两个数值阈值；LED关闭及反应盘外圈属于诊断原理背景。',
        47:'只问次数及通过标准；LED开启属于原理背景，不要求每次复述。',
        48:'问机型间能否通用及编码；对应部件的编码和兼容关系必答，三块/两块的数量为背景。',
        49:'必须先确认版本或通讯方式；请求确认时可不展开全部备份恢复步骤。',
    }
    audits=[]
    proposed=[]
    disputed={3,32,33}
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'))
    bundle=store.active()
    conn=store.connect_read_only(bundle.metadata_path)
    try:
        chunks=load_chunk_rows(conn)
    finally:
        conn.close()
    dataset=load_gold_dataset(source)
    maps={c.case_id:map_case_evidence(c,chunks) for c in dataset.cases}
    for old in cases:
        c=copy.deepcopy(old)
        n=int(c['id'][-3:])
        checks=copy.deepcopy(c['answer_checks'])
        if n==1:
            for x in checks:
                if x['id'] in ('F1','F3'): x['role']='context'
        elif n==2:
            checks[2]['role']='context'
        elif n==3:
            for x in checks:
                if x['role']=='safety': x['role']='context'
        elif n in (14,21,49):
            for x in checks:
                if x['id']!='F1': x['role']='context'
        elif n==24:
            checks=[check('F1','core','定量项目根据校准调整后的主曲线，将样本发光值转换为浓度。'),
                    check('F2','core','定性项目根据校准得到Cutoff，比较样本发光值与Cutoff判断阴阳性。'),
                    check('C1','context','原文定量采用1–3点校准，定性采用1–2点校准。')]
        elif n==27:
            checks[0]['role']='context'
        elif n==32:
            checks[1]=check('F2','core','核查失控是否关联新装载试剂，检查磁珠瓶是否卡到位及是否混匀；重新装载、混匀后复测验证，不直接断言当前故障原因。')
            checks.append(check('C2','context',old['answer_checks'][1]['text']))
        elif n==33:
            checks[1]=check('F2','core','结合复测发光值变化，检查试剂磁珠是否充分混匀；不能仅凭失控方向确定唯一故障。')
            checks.append(check('C1','context','案例最终发现维修后未进入操作软件，试剂仓未自动混匀，磁珠沉降。'))
        elif n==37:
            checks[0]=check('F1','core','先分析并排除异常原因后再校准，不应在未排查原因时立即反复定标。')
            checks.append(check('C1','context','原资料称该现象为校准失败，应确认并记录原因。'))
        elif n==38:
            checks=[check('F1','context',old['answer_checks'][0]['text']),
                    check('F2','core','先核查TBIL、DBIL定标液复溶后的稳定性及见光分解，按说明书确认其存放时间和保存条件。'),
                    check('C1','context','天间校准也会引入额外不确定因素。')]
        elif n==41:
            checks=[check('F1','core','Ca、Mg、Fe先变差时检查水质。'),
                    check('F2','core','LDH明显偏低时检查反应盘温度，尤其是否偏低。'),
                    check('C1','context','Ca、Mg、Fe对水质较敏感；反应温度异常时ALP、ALT、AMY等酶类也可受影响。')]
        elif n==46:
            checks[0]=check('F1','core','暗计数诊断检查53个杯位。')
            checks.append(check('C1','context','在无光、LED关闭时测量反应盘外圈杯位。'))
        elif n==47:
            checks[0]=check('F1','core','PMT重复测试20次，计算DCF因子。')
            checks.append(check('C1','context','诊断时参考模块LED开启。'))
        elif n==48:
            checks=[check('F1','core','CL-6000i反应盘和磁分离盘温度采集板编码均为051-002415-00。'),
                    check('F2','core','CL-8000i反应盘板为051-002415-00，磁分离盘板为051-004328-01；不同编码的这两种板卡不兼容，不能一概通用。'),
                    check('C1','context','原文CL-6000i为三块同编码板，CL-8000i有两块磁分离盘板。')]
        changed=checks!=old['answer_checks']
        c['answer_checks']=checks
        c['required_facts']=[x['text'] for x in checks if x['role'] in ('core','safety')]
        c['review_status']='pending'
        c['annotation_provenance']={'source_version':'v3','source_status':old['review_status'],
            'status':'pending','note':'仅为逐题待审核草案，不替代历史reviewed标签。'}
        proposed.append(c)
        ms=maps[c['id']]
        audit={'id':c['id'],'question':c['question'],'expected_behavior':c['expected_behavior'],
            'missing_fields':c['missing_fields'],'source_review_status':old['review_status'],
            'review_status':'pending','proposal':'disputed' if n in disputed else 'revise' if changed else 'keep',
            'reason':reasons.get(n,'当前检查项直接回答所问内容或约束不应出现的错误结论，建议保留。'),
            'original_checks':old['answer_checks'],'proposed_checks':checks,
            'evidence':[{'evidence_id':m.evidence_id,'file':m.file,'page':m.page,'quote':m.quote,
                         'indexed':bool(m.chunk_ids),'confidence':m.confidence} for m in ms if m.kind=='positive'],
            'scope_note':'复用原已审核引文，本轮只核实其仍存在于当前AI索引；未重新打开全部原文件，未新增业务事实或引文。',
            'check_review':[{'id':x['id'],'role':x['role'],
                'decision':'required' if x['role'] in ('core','safety') else 'optional',
                'reason':'直接回答问题或约束处理路径' if x['role']=='core' else
                         '本题涉及实际拆装，保留操作安全要求' if x['role']=='safety' else
                         reasons.get(n,'原检查项已标为可选背景，保持。')} for x in checks]}
        audits.append(audit)
    # A source discrepancy is an annotation dispute, never an automatic new Gold.
    native_path=Path('E:/免疫资料/高速免疫/10-学习参考资料包/高速免疫场景化培训教材-液路V1.1.pptx')
    with zipfile.ZipFile(native_path) as archive:
        native_text=''.join(ET.fromstring(archive.read('ppt/slides/slide22.xml')).itertext())
    indexed=next(r for r in chunks if Path(r['file_path'])==native_path and r['slide_number']==22)
    assert '145±5' in native_text and 'CL-8000i' in native_text
    assert '145±5' in indexed['text'] and 'CL-8000i' in indexed['text']
    disputed_flow=next(c for c in audits if c['id']=='rag-biz-007')
    disputed_flow['proposal']='disputed'
    disputed_flow['reason']='原Gold机型表为140±5；另一份高速液路V1.1第22张写CL-8000i调试145±5。需业务确认版本/测量口径，不能让用户任选文件，也不自动改变原标准答案。'
    disputed_flow['additional_evidence_for_review']={'file':str(native_path),'slide':22,
        'quote':indexed['text'],'native_and_index_verified':True,
        'note':'仅争议记录，不加入计分正向证据，不修改v3或v4草案的原数值。'}
    assert source.read_bytes()==original
    container={'_阅读说明':{'用途':'v3逐题标注复核草案；全部pending，必须业务人员逐题审核后才能替代正式标签。',
        '依据':'原问题及既有Gold引文，不读取模型答案反推标签。',
        '历史版本':'v1/v2/v3不修改；本草案不是应用默认测试集。',
        'bundle_id':bundle.bundle_id,'source_sha256':hashlib.sha256(original).hexdigest()},
        '_字段说明':{
            'answer_checks.id':'检查项编号；拆分后可增加C编号背景项，不影响题目id。',
            'answer_checks.role':'core=本题必答；safety=本题所涉操作必须满足的安全要求；context=可选背景，遗漏不扣分。',
            'required_facts':'仅列core和safety，与建议检查项语义保持一致。',
            'review_status':'本草案均为pending，未经用户/业务人员审核不得进入正式计分。历史v3保持reviewed。',
            'annotation_provenance':'记录原标注版本和审核状态，不是系统实际运行结果。'},
        'cases':proposed}
    (FIXTURES/'rag_gold_v4_proposed_review.json').write_text(json.dumps(container,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (FIXTURES/'rag_gold_v4_proposed.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in proposed),encoding='utf-8')
    auditdoc={'source_sha256':hashlib.sha256(original).hexdigest(),'bundle_id':bundle.bundle_id,
        'counts':{'total':len(audits),'revise':sum(a['proposal']=='revise' for a in audits),
                  'disputed':sum(a['proposal']=='disputed' for a in audits),'keep':sum(a['proposal']=='keep' for a in audits)},
        'cases':audits}
    (FIXTURES/'rag_gold_v3_rubric_audit.json').write_text(json.dumps(auditdoc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    a=load_gold_dataset(FIXTURES/'rag_gold_v4_proposed.jsonl',reviewed_only=False)
    b=load_gold_dataset(FIXTURES/'rag_gold_v4_proposed_review.json',reviewed_only=False)
    assert [c.raw for c in a.cases]==[c.raw for c in b.cases]
    assert all(c.review_status=='pending' for c in a.cases)
    print(auditdoc['counts'])


if __name__=='__main__':main()
