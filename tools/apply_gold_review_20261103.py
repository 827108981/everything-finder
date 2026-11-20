"""Apply the user's rubric review without rewriting historical labels or runs.

Only questions, Gold evidence and the user's review inform these annotations.
No generated answers or evaluation verdicts are read.
"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import fitz

from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import GoldDatasetError, load_gold_dataset
from local_full_text_search.rag.index_store import RagIndexStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/rag/fixtures'
OUT = ROOT / 'reports/gold_review_20261103'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(text: str) -> str:
    return re.sub(r'\s+', '', text)


def native_slide(path: Path, page: int) -> str:
    with ZipFile(path) as z:
        return ''.join(ET.fromstring(z.read(f'ppt/slides/slide{page}.xml')).itertext())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preserved = [
        *FIXTURES.glob('rag_gold_v[123]*'),
        FIXTURES / 'rag_gold_v4_proposed_review.json',
        FIXTURES / 'rag_gold_v4_proposed.jsonl',
        FIXTURES / 'rag_gold_v3_rubric_audit.json',
    ]
    hashes = {str(p): digest(p) for p in preserved if p.is_file()}
    source = json.loads((FIXTURES / 'rag_gold_v4_proposed_review.json').read_text(encoding='utf-8'))
    cases = copy.deepcopy(source['cases'])
    changes = []
    wording = {
        (8, 'F2'): '通电正常但不转动时，判断蠕动泵损坏，需要更换。',
        (16, 'F2'): '有水时1988≤AD≤2108，无水时AD≤900。',
        (16, 'F3'): '仅凭报警不能确定是光耦、板卡还是供电故障，现有证据不足以判定具体故障件。',
        (19, 'F2'): '排除故障和污染源后灌注40次以上，再重新定标。',
        (22, 'F1'): '可能是底物管路破裂或底物瓶已空，不能仅凭低值确定其中一种。',
        (23, 'F2'): '先灌注底物，再重新定标。',
        (25, 'F1'): 'TSH标本室温保存不超过8小时。',
        (25, 'F2'): 'CEA标本室温保存不超过72小时。',
        (26, 'F1'): 'PRL校准有效期为28天。',
        (26, 'F2'): 'E2校准有效期为14天。',
        (28, 'F1'): '排查试剂是否超过定标周期、底物液是否过期、校准品状态与操作，以及仪器长期维护情况。',
        (29, 'F2'): '按厂家给出的阴性质控范围上限判断，未超过上限可视为在控；具体上限以适用产品资料为准。',
        (30, 'F2'): '初始SD可按上述两种方案估计；仅有20个结果时，将SD扩大30%。',
        (31, 'F1'): '1₂S：一个质控值超过均值±2SD，为警告规则。',
        (31, 'F2'): '1₃S：一个质控值超过均值±3SD，为失控规则。',
        (32, 'F2'): '核查是否更换了新试剂，检查磁珠瓶是否卡到位、磁珠是否混匀；重新装载并混匀后复测质控。',
        (33, 'F1'): '若“＞”集中于竞争法、“＜”集中于夹心法，可能提示发光值偏低。',
        (33, 'F2'): '结合复测发光值变化，检查试剂磁珠是否充分混匀。',
        (36, 'F1'): '线性校准主要适用于比色法，常用两点校准：空白溶液和一个已知浓度校准品。',
        (39, 'F1'): '双波长可减轻黄疸、溶血、乳糜等有色溶液干扰，以及比色杯气泡或灯泡不稳定的影响。',
        (39, 'F2'): '不同组合的抗干扰能力不同，应向试剂厂家确认合适的双波长设置。',
        (40, 'F1'): '小样本量项目先受影响、而大样本量项目较稳定时，优先排查样本针轻微堵塞。',
        (40, 'F2'): '若所有项目都变差，再检查冲洗头堵塞等共性问题。',
        (42, 'F2'): 'R1每测消耗224 μL、约可测803人份；R2每测消耗62 μL、约可测725人份，因此R2先用完。',
        (46, 'F2'): '每个杯位不超过300 CPS为合格，通常正常值不超过200 CPS。',
        (50, 'F1'): '先查UPS供电或零地电压、漏光、反应盘测光位发光异物及光学异常。',
    }
    for c in cases:
        old = copy.deepcopy(c)
        n = int(c['id'][-3:])
        if n == 1:
            # Instrument power does not establish switch power.
            next(x for x in c['answer_checks'] if x['id'] == 'F1')['role'] = 'core'
        if n == 7:
            # Do not keep a disputed unique value as a mandatory answer.
            # The original value remains in historical Gold and the audit.
            c['answer_checks'] = []
        if n == 24:
            c['question'] = '采用4PLC的定量项目和采用COI的定性项目，分别怎样根据样本发光值得到浓度或阴阳性结果？'
            c['constraints']['object'] = '4PLC定量与COI定性结果'
            c['answer_checks'] = [
                {'id': 'F1', 'role': 'core', 'text': '4PLC定量：根据校准调整后的主曲线，将样本发光值转换为浓度。'},
                {'id': 'F2', 'role': 'core', 'text': 'COI定性：由校准得到Cutoff，比较样本发光值与Cutoff判断阴阳性。'},
            ]
            c['forbidden_claims'] = [
                '把4PLC描述为所有定量检测通用的唯一方法。',
                '把COI描述为所有定性检测通用的唯一方法。',
                '把COI的Cutoff比较说成直接输出浓度值。',
            ]
        if n == 27:
            e = c['gold_evidence'][0]
            e['quote'] = '如果两次重测均为阴性，此样本判断为阴性；如果其中至少一次的结果为阳性，此样本判断为阳性'
            e['position'] = '复测样本的结果解释／阴阳性判读'
            e['why_sufficient'] = '直接给出两次均阴性与至少一次阳性的判读规则，不要求复述前处理。'
            c['forbidden_claims'] = ['两次复测一阴一阳时判阴性。', '两次复测均阴性时仍仅凭初测阳性判阳性。']
        optional = [x for x in old['answer_checks'] if x['role'] == 'context']
        c['answer_checks'] = [x for x in c['answer_checks'] if x['role'] != 'context']
        for x in c['answer_checks']:
            x['text'] = wording.get((n, x['id']), x['text'])
        if n == 33:
            c['forbidden_claims'].append('仅凭“＞”和“＜”确定唯一故障原因。')
        c['required_facts'] = [x['text'] for x in c['answer_checks'] if x['role'] in ('core', 'safety')]
        # The user approved other proposals, but has not seen this new wording.
        c['review_status'] = 'disputed' if n == 7 else 'pending' if n == 24 else 'reviewed'
        c['annotation_provenance'] = {
            'source_version': 'v3 / v4_proposed', 'source_status': 'reviewed / pending',
            'status': c['review_status'], 'review_date': '2026-11-03',
            'review_basis': '用户认可其余精简建议，并明确027仅需判读结果；本次不修改历史标签。',
            'note': '024新问法待确认。' if n == 24 else '007原资料同单位但数值不同，未裁定。' if n == 7
            else '按用户审核意见落实；仅保留直接回答问题的检查项，操作顺序和适用安全要求保留。',
        }
        changes.append({'id': c['id'], 'question': c['question'], 'review_status': c['review_status'],
                        'removed_optional_context': optional, 'previous_checks': old['answer_checks'],
                        'current_checks': c['answer_checks'], 'question_changed': old['question'] != c['question']})

    store = RagIndexStore(Path('E:/everythingfinder-db/rag'))
    bundle = store.active()
    assert bundle.bundle_id == source['_阅读说明']['bundle_id']
    conn = store.connect_read_only(bundle.metadata_path)
    try:
        chunks = load_chunk_rows(conn)
    finally:
        conn.close()

    def indexed(path: Path, page: int):
        return next(r for r in chunks if Path(r['file_path']) == path and
                    (r.get('page_number') or r.get('slide_number')) == page)

    fluid = Path('E:/免疫资料/免疫场景化培训教材-液路V1.1.pdf')
    highspeed = Path('E:/免疫资料/高速免疫/10-学习参考资料包/高速免疫场景化培训教材-液路V1.1.pptx')
    with fitz.open(fluid) as doc:
        page19 = doc[18].get_text()
        doc[18].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(OUT / 'fluid_page_19.png')
    header, row = '吸液流量 μL/s', 'CL-8000i\nBM80\n140±5\n使用蠕动泵吸液'
    for text in [page19, indexed(fluid, 19)['text']]:
        assert compact(header) in compact(text) and compact(row) in compact(text)
    ppt_quote = '磁分离吸液流量调试：145±5μL/s（通过参数调整）；'
    for text in [native_slide(highspeed, 22), indexed(highspeed, 22)['text']]:
        assert compact(ppt_quote) in compact(text)
    conflict = {
        'id': 'rag-biz-007', 'status': 'disputed',
        'conclusion': '两处单位均为μL/s；数值分别为140±5和145±5，不是有无单位或单位换算的差别。区间有交集但不等价。',
        'sources': [
            {'file': str(fluid), 'page': 19, 'position': '吸液流量表头及CL-8000i/BM80行',
             'quotes': [header, row], 'native_verified': True, 'index_verified': True,
             'chunk_id': indexed(fluid, 19)['chunk_id']},
            {'file': str(highspeed), 'page': 22, 'position': '磁分离吸液流量调试',
             'quotes': [ppt_quote], 'native_verified': True, 'index_verified': True,
             'chunk_id': indexed(highspeed, 22)['chunk_id']},
        ],
        'need_confirmation': '确认两份文件适用的配置、修订和调试规范；不能自行选用其中一个值作为唯一标准。',
        'scoring': '旧值在历史版本及审核记录中保留，当前必答检查项清空；本题disputed，禁止进入正式计分。原证据保留供追溯，冲突材料不新增为计分Gold。',
        'historical_required_value': '140±5 μL/s；仅追溯，不是本版已裁定的唯一答案。',
    }
    cases[6]['annotation_provenance']['evidence_conflict'] = conflict
    # Recheck the two edited answer scopes in the actual native slides and index.
    native_checks = []
    for n in (24, 27):
        c = cases[n - 1]
        for e in c['gold_evidence']:
            path = Path(e['file'])
            assert compact(e['quote']) in compact(native_slide(path, e['page']))
            r = indexed(path, e['page'])
            assert compact(e['quote']) in compact(r['text'])
            native_checks.append({'id': c['id'], 'file': str(path), 'page': e['page'],
                                  'quote': e['quote'], 'native_verified': True,
                                  'index_verified': True, 'chunk_id': r['chunk_id']})

    container = copy.deepcopy(json.loads((FIXTURES / 'rag_gold_v3_review.json').read_text(encoding='utf-8')))
    container['cases'] = cases
    statuses = dict(Counter(c['review_status'] for c in cases))
    info = container['_阅读说明']
    info.update({
        '用途': 'v4：按2026-11-03用户审核精简Gold，问什么答什么；不修改历史v1/v2/v3或旧实验草案。',
        '状态提示': f'{statuses}。024新问法pending；007数值冲突disputed；其余按用户明确认可标为reviewed。',
        '答案判分版本': 'answer-checks-v2；core仅为所问必需事实，safety仅为适用操作要求；可选背景移到审核记录，不要求复述。',
        '审核提示': '先确认024新问法及007冲突。当前应用默认数据集未切换；包含pending/disputed的整包不能正式加载。',
        '状态变更记录': '2026-11-03：用户认可其余精简建议，027取消离心要求；024改为明确限定4PLC/COI并待确认；007保留争议。',
        '历史对照': '旧实验使用rag_gold_v4_proposed；本文件是审核后修订版，旧实验分数不适用于本文件，未重跑也不宣称算法进步。',
        '原始资料核验范围': '本次重开液路PDF第19页、PPTX第22张及024/027对应原生幻灯片；其余复用历史审核证据，并重新验证当前索引中的定位。',
    })
    container['_审核记录'] = {'date': '2026-11-03', 'source_sha256': hashes,
        'principle': '证据保留原文以供核查；答案检查项归纳所问要点，不照抄整段、不强制背景；不按系统答案反推。'}
    review_path = FIXTURES / 'rag_gold_v4_review.json'
    jsonl_path = FIXTURES / 'rag_gold_v4.jsonl'
    write_json(review_path, container)
    jsonl_path.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases), encoding='utf-8')
    a, b = [load_gold_dataset(p, reviewed_only=False) for p in (review_path, jsonl_path)]
    assert [c.raw for c in a.cases] == [c.raw for c in b.cases]
    assert len(a.cases) == 50 and statuses == {'reviewed': 48, 'disputed': 1, 'pending': 1}
    rejected = []
    for path in (review_path, jsonl_path):
        try:
            load_gold_dataset(path)
        except GoldDatasetError as exc:
            assert 'rag-biz-007' in str(exc) and 'rag-biz-024' in str(exc)
            rejected.append(str(path))
        else:
            raise AssertionError('Unreviewed cases must not enter scoring')
    evidence = []
    for c in a.cases:
        for m in map_case_evidence(c, chunks):
            if m.kind != 'positive':
                continue
            assert m.declared_chunk_ids, (c.case_id, m.evidence_id, m.confidence)
            evidence.append({'id': c.case_id, 'evidence_id': m.evidence_id, 'file': m.file,
                             'page': m.page, 'quote': m.quote, 'declared_chunk_ids': m.declared_chunk_ids,
                             'confidence': m.confidence, 'index_verified': True})
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    validation = {
        'bundle_id': bundle.bundle_id, 'total': 50, 'statuses': statuses,
        'expected_behavior': dict(Counter(c.expected_behavior for c in a.cases)),
        'json_jsonl_identical': True, 'diagnostic_loader': '50 cases pass in both formats',
        'official_loader': 'Correctly rejects 007 disputed and 024 pending',
        'official_load_rejected_paths': rejected, 'historical_files_and_proposed_draft_unchanged': True,
        'question_changes': [c['id'] for c in changes if c['question_changed']],
        'all_required_facts_equal_mandatory_checks': all(list(c.required_facts) ==
            [x['text'] for x in c.answer_checks if x['role'] in ('core', 'safety')] for c in a.cases),
        'positive_evidence_index_verified': len(evidence), 'source_reopened': native_checks,
        'flow_conflict': conflict, 'evidence': evidence,
        'output_sha256': {str(p): digest(p) for p in (review_path, jsonl_path)},
        'algorithm_or_default_changed': False, 'new_evaluation_performed': False,
    }
    write_json(OUT / 'verification.json', validation)
    write_json(OUT / 'changes.json', changes)
    md = ['# Gold v4 审核后修订（2026-11-03）', '',
          '问什么答什么：只检查直接回答本题的必要事实、适用操作安全和必要顺序。原文引文用于核查，不是要求逐字复述的标准答案。可选背景已移入审核记录。', '',
          '本版50题：48 reviewed，024 pending，007 disputed。用户已认可其他精简建议。历史版本、旧实验草案、应用默认数据集均保留；未重新评测。', '',
          '024新问法明确限定4PLC和COI，不将其泛化为所有定量/定性方法；新措辞待确认。027只判结果，不要求离心条件。001仍须核查交换机供电：仪器上电不代表交换机已上电。', '',
          '007：液路PDF第19页表头单位为μL/s，CL-8000i/BM80行为140±5；高速液路PPTX第22张为145±5 μL/s。两者同单位，数值不同，仍需确认适用规范。旧值仅保留追溯，本题不计分。', '',
          '两种格式50条完全一致，诊断加载通过；正式加载正确拦截007和024。69条正向引文均在当前AI包的声明文件和页码命中。024、027和007两处冲突原文已重新打开核查，其余使用历史审核证据，未声称本次重新逐页核查所有文件。', '',
          'AI bundle_id：`' + bundle.bundle_id + '`。旧版实验结果不能直接当作本版成绩；改标注造成的分数变化不能算作算法进步。', '',
          '## 字段说明', '',
          '- `question`：真实用户问法；`group_id`：同一业务问题共用组名。',
          '- `expected_behavior`：answer回答、partial_answer部分回答、clarify澄清、refuse拒答。',
          '- `constraints`：问题中明确的范围、阶段、方向和对象，未说的留空。',
          '- `answer_checks`：core必答事实，safety适用操作安全；不要求背景复述。`required_facts`与必答检查项一致。',
          '- `gold_evidence`：文件、从1开始的页码/幻灯片、位置、原文及支持理由；用于验证根据。',
          '- `hard_negatives`：看似相似却不适用的证据及原因。',
          '- `forbidden_claims`：具体不能出现的错误结论。`missing_fields`：必须澄清的缺失条件。',
          '- `corpus_snapshot`：已发布AI索引包编号。`review_status`：reviewed可计分，pending待确认，disputed有争议。',
          '- `annotation_provenance`：审核来源、日期和修订说明；不是系统实际回答。', '',
          '## 逐题审核清单', '', '| 题号 | 问题 | 必需回答要点 | 状态 |', '|---|---|---|---|']
    for c in cases:
        facts = '；'.join(c['required_facts']) or '拒答，不编造结论或更换件。'
        if c['id'] == 'rag-biz-007':
            facts = '存在140±5与145±5 μL/s两种记载，暂不确定唯一标准，不计分。'
        md.append('| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in
                  (c['id'], c['question'], facts, c['review_status'])) + ' |')
    md += ['', '详细原文核查：`reports/gold_review_20261103/verification.json`；可选背景及修改前后检查项：同目录`changes.json`。', '']
    (FIXTURES / 'rag_gold_v4.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({k: validation[k] for k in ('total', 'statuses', 'json_jsonl_identical',
        'expected_behavior', 'positive_evidence_index_verified', 'historical_files_and_proposed_draft_unchanged')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
