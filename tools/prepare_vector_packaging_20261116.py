"""Freeze source-derived pending acceptance questions before candidate scoring."""
from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
from collections import Counter
from pptx import Presentation
from docx import Document
import fitz
from tools.validate_selection_factorial import ROOT, sha, write
from tools.diagnose_vector_recall import ro
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset

OUT = ROOT / 'reports/rag_vector_packaging_20261116'
FIXTURE = ROOT / 'tests/rag/fixtures/rag_vector_source_holdout_20261116.jsonl'
DB = Path('E:/everythingfinder-db/rag/rag.g000033-71ec3c52f3/metadata.db')

def fold(s): return re.sub(r'[\s|]+', '', s).casefold()

def original(path, page):
    p = Path(path)
    if p.suffix.lower() == '.pptx':
        s = Presentation(p).slides[page-1]
        parts = []
        def visit(shapes):
            for sh in shapes:
                if sh.has_text_frame: parts.append(sh.text)
                if sh.has_table:
                    parts.extend('\n'.join(c.text for c in row.cells) for row in sh.table.rows)
                if hasattr(sh, 'shapes'): visit(sh.shapes)
        visit(s.shapes)
        if s.has_notes_slide:
            parts.append(s.notes_slide.notes_text_frame.text)
        return '\n'.join(parts)
    if p.suffix.lower() == '.pdf':
        with fitz.open(p) as d: return d[page-1].get_text()
    d = Document(p)
    extras = [text for section in d.sections for container in (section.header, section.footer)
              for text in [*(p.text for p in container.paragraphs), *('\n'.join(c.text for c in r.cells) for t in container.tables for r in t.rows)]]
    return '\n'.join([*(p.text for p in d.paragraphs), *('\n'.join(c.text for c in r.cells) for t in d.tables for r in t.rows), *extras])

# Source choice and questions are manually authored from original documents, not retrieval output.
DEFINITIONS = [
 ('CL-8000i超声频率异常302340300.pdf', [
  (1, 'CL-8000i报302340300超声频率异常，什么情况下会触发？', '如果频率有效值在【正常空化频率下限，正常空化频率上限】区间外', '混匀结束后，频率有效值超出正常空化频率上下限区间时报警。', 'condition'),
  (1, 'CL-8000i超声频率异常报警后，当前这次测试会作废吗？', '警告，当前单个测试作废', '报警为警告，当前单个测试作废。', 'condition')]),
 ('CL-8000i超声空化电压高302340400.pdf', [
  (1, 'CL-8000i超声空化电压高报警是在什么情况下触发的？', '没有发生频率异常，空化电压个数正常，但空化电压高于报警内显示的阈值的个数多于预期', '无频率异常且电压个数正常，但高于报警阈值的电压个数多于预期。', 'condition'),
  (1, 'CL-8000i超声空化电压高，处理策略设为2时测试怎么处理、结果带什么标记？', '2：测试保留，日志标注Ult.NA，测试结果标记Ult.NA', '测试保留；日志记录Ult.NA，结果带Ult.NA标志。', 'table')]),
 ('BS-2800-213120802光源灯不稳定V0.2.docx', [
  (None, 'BS-2800M报光源灯不稳定，环境温度要检查什么？', '检查环境温度，是否在15-30°，并且无明显波动', '检查环境是否15～30℃且无明显温度波动。', 'numeric'),
  (None, 'BS-2800M的50W光源灯组件，下单编码是什么？', '50W光源灯组件', '50W光源灯组件编码115-014942-00。', 'table')]),
 ('BS-2800-215050602主真空积液V0.2.docx', [
  (None, 'BS-2800M主真空积液报警是检测到什么信号、持续多久触发的？', '连续10S检测到FL12的ON信号', '连续10秒检测到FL12的ON信号时报警。', 'numeric'),
  (None, 'BS-2800M主真空容器组件的下单编码是什么？', '主真空容器组件', '主真空容器组件编码115-074333-00。', 'table')]),
 ('BS-1000M-235150702-反应杯清洗水温度超出范围，原因：温度过低.docx', [
  (None, 'BS-1000M反应杯清洗水温度报警，清洗水应保持在什么温度范围？', '清洗水温度可以保持在35.5-45.5℃', '清洗水温度应保持35.5～45.5℃。', 'numeric'),
  (None, 'BS-1000M清洗水温度采集板VPP测试点的正常电压范围是多少？', '正常范围：12V±5%，即11.4~12.6V', 'VPP正常12V±5%，即11.4～12.6V。', 'numeric')]),
 ('用服工具使用说明.pdf', [
  (7, '用服工具的跑水测试，维护、维修和装机样本编号怎么区分？', '维护：WATER-M-**  (如：WATER-M-01）\n维修：WATER-R-**\n装机：WATER-S-**', '维护WATER-M-**；维修WATER-R-**；装机WATER-S-**。', 'table'),
  (7, '跑水测试失败，应该查哪个日志、搜索哪个字段看失败原因？', '打开Log->TraceLog00.csv, 查找“WaterTestJudgeHelper”字段', '打开Log下TraceLog00.csv，搜索WaterTestJudgeHelper字段。', 'identifier')]),
 ('IVD工程师基础技能-热工-有声V1.1.pptx', [
  (6, '去离子水加热保护开关和反应盘加热保护开关，断开与闭合温度分别是多少？', '去离子水加热保护开关', '示例中去离子水开关60℃断开，反应盘开关50℃断开；两者低于30℃闭合。', 'table'),
  (7, '风扇有三根线，就一定有转速反馈吗？', '不是所有三根线的都具有反馈信号，要看最终接到板卡上的是几根线才能确定', '不一定，要检查最终接到板卡的线数。', 'condition')]),
 ('IVD工程师基础技能-电脑-有声V1.1.pptx', [
  (5, '仪器电脑自带的系统还原镜像一般放哪个盘，文件后缀是什么？', '仪器操作电脑自带的系统通常会放在操作电脑的E盘，文件后缀为.gho', '通常放E盘，后缀.gho。', 'identifier'),
  (5, '还原仪器电脑系统时，目标磁盘该选什么类型，通常还原哪个分区？', '选择需要还原的目标磁盘（必须选择本地磁盘）', '目标选本地磁盘，分区通常为C盘。', 'procedure')]),
 ('IVD工程师基础技能-硬件-有声V1.1.pptx', [
  (5, '光耦与板卡断开和接好时，红黑供电线之间的正常电压分别是多少？', '此时电压正常为5V左右', '断开时板卡侧约5V，接好后约1.2～1.3V。', 'condition'),
  (6, 'DB44串口线的44针端和25针端，分别接电脑还是生化仪？', '接电脑端-串口公头（44针）\n接生化仪端-串口公头（25针）', '44针接电脑，25针接生化仪。', 'table')]),
 ('IVD工程师基础技能-液路及气路-有声V1.1 .pptx', [
  (7, '浮子传感器怎么用万用表判断好坏？', '浮子在原位置时，万用表蜂鸣器不响\n3、手动抬起浮子，此时万用表蜂鸣器响起', '蜂鸣档测接头外壁和内芯；原位置不响，抬起后响为正常。', 'procedure'),
  (9, '三通电磁阀断电和通电时，COM分别与NO还是NC相通？', '电磁阀通电：NO与COM不相通，NC与COM相通', '断电COM通NO不通NC；通电COM通NC不通NO；NO与NC不相通。', 'condition')]),
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        print('Already frozen; refusing to reselect questions'); return
    with ro(DB) as c: rows = load_chunk_rows(c)
    manifestpath = ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json'
    manifest = json.loads(manifestpath.read_text(encoding='utf8'))
    oldcases = [c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'], reviewed_only=False).cases]
    oldmaps = [m for c in oldcases for m in map_case_evidence(c, rows)]
    byid = {r['chunk_id']: r for r in rows}
    oldpaths = {byid[p.chunk_id]['file_path'] for m in oldmaps for a in m.alternatives for p in a.parts}
    new = []; audit = []; sourcehashes = {}
    for group, (name, defs) in enumerate(DEFINITIONS, 1):
        paths = {r['file_path'] for r in rows if Path(r['file_path']).name == name}
        assert len(paths) == 1, (name, paths)
        path = next(iter(paths)); assert path not in oldpaths
        sourcehashes[path] = sha(path)
        for page, question, anchor, fact, kind in defs:
            candidates = [r for r in rows if r['file_path'] == path and (page is None or (r['slide_number'] or r['page_number']) == page) and fold(anchor) in fold(r['text'])]
            assert candidates, (name, page, anchor)
            parent = candidates[0]
            raw = original(path, page)
            assert fold(anchor) in fold(raw), (name, page, 'original anchor missing')
            # Freeze the complete relevant parent quote for multi-field/row facts.
            # Evaluation remains source and location bound; each fact is stated separately.
            quote = anchor
            # Multi-field questions need their actual fields, not an anchor
            # that merely locates the surrounding page.
            if kind == 'table' or len(fact.split('；')) > 1 or kind == 'procedure' or '断开和接好' in question:
                quote = parent['text'].strip()
                # DOCX contains several paragraphs/tables; its full native parser representation
                # is available as original provenance, not a fabricated physical page number.
                lines = []
                for line in quote.splitlines():
                    if line.strip() and fold(line) not in fold(raw):
                        break  # Stop before drawing-only OCR/unsupported footer, never invent it.
                    lines.append(line)
                quote = '\n'.join(lines).strip()
                assert fold(anchor) in fold(quote), (name, 'anchor outside verified prefix')
            if name.endswith('光源灯不稳定V0.2.docx') and kind == 'table':
                quote = next(line for line in parent['text'].splitlines() if '50W光源灯组件 |' in line)
            if name.endswith('主真空积液V0.2.docx') and kind == 'table':
                quote = next(line for line in parent['text'].splitlines() if '主真空容器组件 |' in line)
            if '热工-有声' in name and kind == 'table':
                quote = parent['text'].split('不同温度保护开关的断开温度和闭合温度')[0].strip()
            if '断开和接好' in question:
                quote = parent['text'].split('3. 光耦遮挡')[0].strip()
            if 'DB44' in question:
                quote = anchor
            ident = f'vector-source-independent-{len(new)+1:03}'
            record = dict(id=ident, group_id=f'vector_source_{group:02}_{kind}_{len(new)+1}', question=question,
                corpus_snapshot=manifest['bundle_id'], expected_behavior='answer',
                constraints=dict(scope='', stage='', direction='', object=''), missing_fields=[],
                gold_evidence=[dict(file=path, page=page, position=parent['location'], quote=quote, relevance=2,
                                    why_sufficient='原资料及当前索引双重核对；具体对象、条件、数值在该位置。')],
                hard_negatives=[], required_facts=[fact], forbidden_claims=['将该来源局部示例、型号或操作条件无依据扩展为所有仪器。'],
                answer_checks=[dict(id='F1', role='core', text=fact)], review_status='pending',
                annotation_provenance=dict(source_group=f'source-{group:02}', source_sha256=sourcehashes[path],
                    candidate_scores_seen=False, layout=kind, original_verified=True, index_verified=True,
                    independence='来源未被当前91题Gold及批准替代来源引用；不声称与全部历史调试知识语义独立'))
            if '235150702' in name:
                record['annotation_provenance']['source_discrepancy'] = '文件名报警码235150702与正文235150202不一致；本题只问已明确的型号和参数，不使用报警码判断。'
            new.append(record)
            audit.append(dict(id=ident, file=path, page=page, position=parent['location'], parent=parent['chunk_id'],
                              original_anchor_found=True, index_anchor_found=True, source_not_in_existing_gold=True, quote=quote))
    assert len(new) == 20 and len(sourcehashes) == 10
    assert len(set(sourcehashes.values())) == 10
    oldhashes = {sha(p) for p in oldpaths if Path(p).is_file()}
    assert not oldhashes.intersection(sourcehashes.values()), 'byte duplicate of an existing Gold source'
    FIXTURE.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in new), encoding='utf8')
    loaded = load_gold_dataset(FIXTURE, reviewed_only=False)
    maps = [map_case_evidence(c, rows) for c in loaded.cases]
    assert all(any(m.alternatives for m in ms) for ms in maps)
    write(OUT/'holdout_evidence_audit.json', dict(cases=audit, formats=dict(Counter(Path(p).suffix for p in sourcehashes)),
          duplicate_bytes_with_old_sources=False, old_source_count=len(oldpaths), new_source_count=10,
          limitations=['业务人员尚未逐题复核，全部pending；只用于诊断验收，不计正式分。',
                      '模板相同的超声故障文件存在通用段落重叠；题目只使用各自不同的报警条件和结果策略。',
                      'BS-1000M文件名与正文报警码不一致，参数题不依赖该报警码；原始冲突保留。']))
    protected = [manifestpath, *[ROOT/d['path'] for d in manifest['datasets']], FIXTURE, DB, SETTINGS_PATH,
                 Path('E:/everythingfinder-db/rag/active.json'), Path('E:/everythingfinder-db/rag/embedding_cache.db'),
                 *Path('E:/everythingfinder-db/rag/vector').glob('*/active.json'),
                 Path('E:/everythingfinder-db/rag/vector/g000033-71ec3c52f3/e13c72e6e2e7d83e824c-ad1bda66/vector_units.db')]
    write(OUT/'protocol.json', dict(bundle=manifest['bundle_id'], original_cases=91, holdout_cases=20,
          holdout_fixture=str(FIXTURE), source_hashes=sourcehashes, protected={str(p):sha(p) for p in protected if p.is_file()},
          plan_sha256=sha(ROOT/'docs/rag_vector_representation_packaging_development_validation_plan_20261116.md'),
          factors=['W0','W1','W2','W3','W4','W5'], background_budget=32, max_tokens=192,
          query_instruction_unchanged=True, full_corpus=True, api_calls=0, adapter_enabled=False,
          stop='One complete frozen pass; quality failures do not trigger tuning. Implementation errors can be repaired with logged reruns.',
          adoption='Protect all 46 formal complete and 40 diagnostic facts; >=2 independent source groups gain; no verified new object/scope error.',
          dual_gate='Only if W4 and W5 have verified complementary full-corpus gains; equal total ANN window budgets.',
          source_holdout_frozen_before_candidate_scoring=True))
    print('FROZEN: 91 existing + 20 pending, 10 new sources, 3 formats', flush=True)

if __name__ == '__main__': main()
