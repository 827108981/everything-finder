"""Read-only experiment audit and human-readable 021 report; no model calls."""
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset, _load_rows, GoldDatasetError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/rag_021_method_20261106'
FIX = ROOT / 'tests/rag/fixtures'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf8'))


def rows(name):
    return [json.loads(s) for s in (OUT / name).read_text(encoding='utf8').splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canon(value):
    return re.sub(r'\s+', '', value)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    original = read('resolved_answers.json')
    failures = [r for r in original if r.get('error')]
    recovered = rows('saved_context_recovery_answers.jsonl')
    assert len(failures) == 3 and all('GenerationRequest' in r['error'] for r in failures)
    assert len(recovered) == 3 and not any(r.get('error') for r in recovered)
    answers = sorted([r for r in original if not r.get('error')] + recovered,
                     key=lambda r: (r['arm'], r['repeat']))
    judges = rows('judgements.jsonl') + rows('saved_context_recovery_judgements.jsonl')
    assert len(answers) == 8 and len(judges) == 16
    assert len({(r['arm'], r['repeat']) for r in answers}) == 8
    assert not any(r.get('error') for r in judges)
    dump('final_answers.json', answers)
    dump('final_judgements.json', judges)
    index = {(r['arm'], r['repeat'], r['gold_version']): r['judgement'] for r in judges}

    old = json.loads((FIX / 'rag_gold_v4_1_review.json').read_text(encoding='utf8'))
    new = json.loads((FIX / 'rag_gold_v4_2_review.json').read_text(encoding='utf8'))
    changed = [a['id'] for a, b in zip(old['cases'], new['cases']) if a != b]
    assert changed == ['rag-biz-021']
    equal = _load_rows(FIX / 'rag_gold_v4_2_review.json')[0] == _load_rows(FIX / 'rag_gold_v4_2.jsonl')[0]
    assert equal
    allcases = load_gold_dataset(FIX / 'rag_gold_v4_2_review.json', reviewed_only=False)
    try:
        load_gold_dataset(FIX / 'rag_gold_v4_2_review.json')
        raise AssertionError('Mixed-status file must be rejected by reviewed_only=True')
    except GoldDatasetError as exc:
        strict_rejection = str(exc)
        assert 'rag-biz-007' in strict_rejection and 'rag-biz-024' in strict_rejection
    with tempfile.TemporaryDirectory(prefix='rag_021_loader_') as directory:
        subset = Path(directory) / 'reviewed_only.jsonl'
        subset.write_text(''.join(json.dumps(c.raw, ensure_ascii=False) + '\n'
                                  for c in allcases.cases if c.review_status == 'reviewed'), encoding='utf8')
        formal = load_gold_dataset(subset)
    status = Counter(c.raw['review_status'] for c in allcases.cases)
    behavior = Counter(c.raw['expected_behavior'] for c in formal.cases)
    assert len(allcases.cases) == 50 and len(formal.cases) == 48
    assert all(a['review_status'] == b['review_status'] for a, b in zip(old['cases'], new['cases']))
    unchanged = {p: sha(p) == h for p, h in read('protocol.json')['hashes'].items()}
    assert all(unchanged.values())

    gold = next(c for c in new['cases'] if c['id'] == 'rag-biz-021')
    source_audit = read('original_source_audit.json')
    mappings = read('gold_delta.json')['positive_mappings']
    evidence = []
    for quote, mapping, source in zip(gold['gold_evidence'], mappings, source_audit['quotes']):
        assert quote['quote'] == source['quote']
        evidence.append({'page': quote['page'], 'quote': quote['quote'],
                         'in_original': source['in_original'],
                         'in_active_index': any(a['confirmed'] for a in mapping['alternatives']),
                         'chunk_ids': mapping['chunk_ids']})
    assert len(evidence) == 6 and all(e['in_original'] and e['in_active_index'] for e in evidence)
    native = [r for r in answers if r['arm'] == 'native_current']
    context_audit = []
    for r in native:
        entry = next(e for e in r['entries'] if e['source_id'] == 'S3')
        checks = []
        for e in evidence:
            hit = [x['source_id'] for x in r['entries'] if canon(e['quote']) in canon(x['included_text'])]
            checks.append({'page': e['page'], 'quote': e['quote'], 'source_ids': hit})
        assert not entry['truncated'] and '第 30 ' in entry['location']
        # Page28 cause list is not required when definition and all method actions are present.
        assert all(c['source_ids'] for c in checks if str(c['page']) != '28')
        assert '[S3]' in r['text']
        context_audit.append({'repeat': r['repeat'], 'context_sha256': hashlib.sha256(r['context'].encode()).hexdigest(),
                              'quote_in_actual_included_text': checks, 'page30_truncated': entry['truncated'],
                              'answer_cites_page30': True, 'source_ids': re.findall(r'\[(S\d+)\]', r['text'])})
        dump(f'native_repeat_{r["repeat"]}_trace.json', r)
    assert len({r['context_sha256'] for r in context_audit}) == 1
    baseline = next(r for r in answers if r['arm'] == 'historical_baseline')
    context_equal = all(r['context'] == baseline['context'] for r in answers if r['arm'] in ('native_current', 'saved_context'))
    dump('evidence_audit.json', {'original_and_index': evidence, 'native_actual_context': context_audit,
                                'native_saved_historical_baseline_context_exact_equal': context_equal})

    # Analyst review is stored separately; it never overwrites the automatic judge or Gold.
    analyst = []
    for r in answers:
        j = index[r['arm'], r['repeat'], 'v4.2_answer']
        if r['arm'] == 'historical_H8':
            result, note = 'fail', '开头不分原因要求同时换试剂和校准品，且把LVC写成前次校准自身超限；后文分支不能消除前文误导。'
        elif r['arm'] == 'historical_baseline':
            result, note = 'method_pass_definition_disputed', '两个条件分支与不能仅凭报警选路均正确。定义只说偏差很大，自动判F1缺失；本次native第2/3次相近表述却获通过，存在尺度不一致。保留自动incomplete，不擅自改分。'
        else:
            result, note = 'pass', '两个条件分支、先区分异常来源均已表达；未强制先追问、未无条件沿用或同时更换。原文第30张在实际上下文且被引用。'
        analyst.append({'arm': r['arm'], 'repeat': r['repeat'], 'automatic_verdict': j['verdict'],
                        'analyst_result': result, 'note': note})
    dump('analyst_review.json', analyst)
    readable = ['# 021逐次实际回答与判分', '', '实际回答保留原文；自动判分与分析复核分别记录，不覆盖历史结果。', '']
    for r in answers:
        o = index[r['arm'], r['repeat'], 'v4.1_clarify']
        n = index[r['arm'], r['repeat'], 'v4.2_answer']
        review = next(x for x in analyst if x['arm'] == r['arm'] and x['repeat'] == r['repeat'])
        readable += [f'## {r["arm"]} 第{r["repeat"]}次', '',
                     f'旧Gold：{o["verdict"]}；新Gold：{n["verdict"]}。', '', r['text'], '',
                     f'分析复核：{review["note"]}', '']
        if r['arm'] == 'native_current':
            path = (OUT / f'native_repeat_{r["repeat"]}_trace.json').as_posix()
            readable += [f'[本次原始trace]({path})', '']
    (OUT / 'answers_review.md').write_text('\n'.join(readable), encoding='utf8')
    verification = {'valid_fresh_answers': 6, 'historical_answers': 2, 'judge_outputs': 16,
                    'fresh_answer_errors_after_execution_recovery': 0, 'judge_errors': 0,
                    'pre_request_execution_failures_preserved': 3, 'other_49_exact_equal': True,
                    'only_changed_case': changed, 'all_review_statuses_unchanged': True,
                    'json_jsonl_equal': equal, 'loader_all': len(allcases.cases), 'loader_reviewed': len(formal.cases),
                    'strict_mixed_file_rejection_expected': strict_rejection,
                    'review_status_counts': dict(status), 'reviewed_behavior_counts': dict(behavior),
                    'original_inputs_unchanged': all(unchanged.values()), 'checked_original_file_count': len(unchanged),
                    'native_saved_baseline_context_exact_equal': context_equal,
                    'gold_original_index_quotes_verified': len(evidence)}
    dump('verification.json', verification)

    report = ['# 021方法型问题：Gold修订与隔离验证', '',
              '日期：2026-11-06。结论：按用户确认的“解释条件分支”口径，当前完整流程3/3、固定上下文新生成3/3通过。021不应继续作为“未先追问”的失败样例。本次未改算法；分数变化来自标注纠正，不能叫作算法提升。', '',
              '## 1. 修订范围与可复查输入', '',
              '- 新版本：`tests/rag/fixtures/rag_gold_v4_2_review.json`与同内容JSONL。只改021；其余49题逐字段相同，所有审核状态不变。历史v4.1及以前版本保留，应用默认测试集未切换。',
              '- 021仍为用户已审核的reviewed；expected_behavior由clarify改answer，missing_fields清空。',
              '- core只检查LVC含义、上次异常分支、本次异常分支、不能仅凭报警选路。具体项目需核对数据列为可选context；缺少此句不扣分。',
              '- 禁止仅凭LVC认定上次错、本次物料坏、同时更换两种物料；禁止以必须先追问阻断方法解释。',
              '- 原问题文字未变：“一个项目定标报 LVC，应该沿用上次定标，还是更换本次试剂和校准品？”没有把提示写进问题以引导回答。',
              '- AI索引：`g000033-71ec3c52f3`；资料：`E:\\生化免疫临床知识\\化学发光定标处理思路分析.pptx`第27、28、30张（从1开始）。',
              '- 回答模型gpt-5.6-sol。当前AnswerService、实际配置、检索/生成提示及冻结H8方案均未修改。回答请求不含Gold；判分阶段才传入Gold。', '',
              '## 2. 结果：同一答案分别按新旧Gold重判', '',
              '| 样本 | 旧v4.1 | 新v4.2 | 说明 |', '|---|---|---|---|']
    for arm, label in [('native_current', '当前完整流程'), ('saved_context', '固定已有上下文'), ('historical_baseline', '11/5历史baseline'), ('historical_H8', '11/5历史H8')]:
        cohort = [r for r in answers if r['arm'] == arm]
        oldcounts = Counter(index[r['arm'], r['repeat'], 'v4.1_clarify']['verdict'] for r in cohort)
        newcounts = Counter(index[r['arm'], r['repeat'], 'v4.2_answer']['verdict'] for r in cohort)
        fmt = lambda d: '、'.join(f'{k} {v}' for k, v in d.items())
        note = {'native_current': '3次真实检索至生成；全部两个条件分支正确', 'saved_context': '3次只重做生成；无新增材料或专用提示',
                'historical_baseline': '方法分支正确，F1定义精度被判不足；见争议项', 'historical_H8': '开头无条件更换二者，仍为错误'}[arm]
        report.append(f'| {label} | {fmt(oldcounts)} | {fmt(newcounts)} | {note} |')
    report += ['', '这6次是同一题的重复生成，不是6道独立业务题，也不是新的盲测集。本次不重算或外推48题总体成绩，不以6/6声称普遍可靠。', '',
               '## 3. 实际送给模型的证据', '',
               f'三次完整流程上下文完全一致；与11/5 baseline及固定上下文组也完全一致：{context_equal}。没有注入Gold或新增行级内容。', '',
               '| 资料张号 | 核对原文 | 原PPT | 当前索引 | 本次实际上下文 |', '|---|---|---|---|---|']
    for e in evidence:
        hits = next(c['source_ids'] for c in context_audit[0]['quote_in_actual_included_text'] if c['quote'] == e['quote'])
        q = e['quote'].replace('\n', ' / ')
        report.append(f'| {e["page"]} | {q} | 已找到 | 已找到 | {", ".join(hits) if hits else "未选入"} |')
    report += ['', '第28张可能原因清单未选入；第27张定义与第30张完整分支已选入，足以支持本题。三次均实际引用S3（第30张），该片段未截断。没有把同文件其他页或整页里未发送的内容算成模型看到的证据。', '',
               '## 4. 本次完整流程第2次实际回答', '',
               next(r['text'] for r in native if r['repeat'] == 2), '',
               '## 5. 尚存问题与结论边界', '',
               '1. 历史H8答案仍错误：先说“更换校准品和试剂后重新定标”，再补充条件分支。前面的无条件结论会误导操作，不能因为后面列了正确分支就通过。此次未重新启用或测试H8方案。',
               '2. 历史baseline自动判incomplete，仅因“偏差很大”没有明确写“超出允许范围”；其两条操作路径均正确。本次native第2、3次相似表述却被判covered，说明定义细节的自动判分有尺度波动。原判分保留，另存analyst_review.json，不自行替换正式分数。',
               '3. 可选C1有时显示missing/findings，但角色为context，没有使新回答失败；它不是必须追问项。',
               '4. 021可以从“必须修复追问”的清单中移除。当前证据足以回答方法；本实验没有新增计算工具或算法改动的必要。是否决定某个现场项目具体走哪条路，仍需核查实际情况。',
               '5. 重复生成只验证当前021口径，不证明其他方法型问题、其他报警和追问场景一并解决。若后续做算法修改，应另选未调试的方法型问题验证，不能继续用021调参后宣称泛化。', '',
               '## 6. 文件和执行完整性', '',
               f'- JSON与JSONL逐记录一致：通过；加载全部{len(allcases.cases)}题、正式{len(formal.cases)}题。审核状态：{dict(status)}；正式预期行为：{dict(behavior)}。',
               '- 加载器不是自动过滤非reviewed记录：完整文件用reviewed_only=False通过；默认严格模式按预期拒绝007/024。先筛出48条reviewed后再用严格模式校验通过。未修改加载器或这两题状态。',
               f'- {len(unchanged)}个原业务代码/配置/历史Gold/索引清单/原资料文件哈希前后不变。核验明细见protocol.json和verification.json。',
               '- 6条Gold引文均在原PPT和当前AI索引核验。实际上下文及引用的逐条核对见evidence_audit.json。',
               '- 执行中曾有启动脚本缺少copy导入，以及3个固定上下文作业缺少GenerationRequest导入；均在模型请求前失败。修复实验脚本后只补跑这3个未发送请求的作业，没有丢弃任何有效回答或按质量重试。原错误与恢复记录保留。',
               '- completion.json为首次执行原始统计，包含3个脚本错误；恢复后的最终统计见verification.json。不要把原fresh_answers=6的计划数误认为原执行已成功6次。',
               '- 当前主流程使用实际89秒请求超时；固定上下文诊断和判分请求为180秒。未把请求耗时作为效果指标，未改保存配置。', '',
               '## 7. 复查入口', '',
               '- final_answers.json：6次新回答和2份历史回答，保留实际上下文。',
               '- answers_review.md：按次展开的实际回答、新旧判分与分析复核，方便直接阅读。',
               '- final_judgements.json：16份判分（8份答案×两套Gold），不覆盖原判分。',
               '- native_repeat_1_trace.json、native_repeat_2_trace.json、native_repeat_3_trace.json：逐次完整事件、实际回答、实际上下文、引用审计。',
               '- original_source_audit.json、gold_delta.json、evidence_audit.json：原资料、AI索引及实际引文核对。',
               '- analyst_review.json：独立语义复核，尤其历史baseline定义精度争议和H8矛盾结论。',
               '- protocol.json、recovery_protocol.json、verification.json：参数、原文件哈希、执行恢复和最终校验。', '']
    text = '\n'.join(report)
    (OUT / 'report.md').write_text(text, encoding='utf8')
    (ROOT / 'docs/rag_021_method_validation_20261106.md').write_text(text, encoding='utf8')
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    main()
