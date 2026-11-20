"""Readable full validation report; formal and pending cohorts kept separate."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import statistics

from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import aggregate, STAGE_ORDER, STAGE_LABELS
from local_full_text_search.rag.evaluation.reporting.service import generate_report
from local_full_text_search.rag.evaluation.reporting.timing import case_timing

ROOT = Path(__file__).resolve().parents[1]
GROUP_LABELS = {'formal48': '主测试集：48题正式', 'pending024': '024：待审核诊断',
                'regression10_pending': '回归集：10题待审核诊断',
                'original18_eligible17': '原18题：17题可计分', 'added32_eligible31': '新增32题：31题可计分'}


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def pct(v):
    return '—' if v is None else f'{100 * v:.1f}%'


def summary(rows):
    executed = [r for r in rows if r.get('execution_status') != 'blocked_not_started']
    answered = [r for r in executed if r.get('answer', {}).get('text')]
    judged = [r for r in answered if (r.get('gold_judgement') or {}).get('verdict') not in (None, 'unavailable')]
    v = aggregate(executed)
    v['heuristic_behavior_accuracy'] = v.pop('behavior_accuracy')
    v['heuristic_behavior_confusion'] = v.pop('behavior_confusion')
    v['behavior_measurement_note'] = '原始行为识别依赖固定短语，可能误标自然语言拒答。主报告使用独立语义裁判判断行为，字面启发式仅供诊断。'
    v.update(planned=len(rows), answered=len(answered), judged=len(judged),
        states=dict(Counter(r['execution_status'] for r in rows)),
        verdicts=dict(Counter((r.get('gold_judgement') or {}).get('verdict', 'not_judged') for r in rows)),
        behavior_judge=dict(Counter((r.get('gold_judgement') or {}).get('behavior', 'not_judged') for r in rows)),
        model_judge_consistent_rate=sum(r['gold_judgement']['verdict'] == 'consistent' for r in judged) / len(judged) if judged else None,
        model_judge_consistent_count=sum(r['gold_judgement']['verdict'] == 'consistent' for r in judged),
        behavior_judge_match_count=sum(r['gold_judgement'].get('behavior') == 'match' for r in judged),
        mean_case_ms=statistics.mean(r['duration_ms'] for r in executed) if executed else None)
    calls = Counter()
    modules = Counter()
    for r in executed:
        calls.update(r.get('request_counts', {}))
        modules.update(case_timing(r)['modules_ms'])
    v['request_counts'] = dict(calls)
    v['module_timing_sum_ms'] = dict(modules)
    answer_ms = [next((e['elapsed_ms'] for e in r.get('events', []) if e.get('stage') == 'gold_judging'), None) for r in answered]
    measured = [x for x in answer_ms if x is not None]
    judge_ms = [r['duration_ms'] - a for r, a in zip(answered, answer_ms) if a is not None]
    v['mean_answer_ms'] = statistics.mean(measured) if measured else None
    v['mean_judge_ms'] = statistics.mean(judge_ms) if judge_ms else None
    v['by_expected_behavior'] = {}
    for behavior in ('answer', 'partial_answer', 'clarify', 'refuse'):
        subset = [r for r in judged if r['expected_behavior'] == behavior]
        if subset:
            v['by_expected_behavior'][behavior] = {
                'judged': len(subset),
                'consistent': sum(r['gold_judgement']['verdict'] == 'consistent' for r in subset),
                'behavior_match': sum(r['gold_judgement']['behavior'] == 'match' for r in subset)}
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--skip-native', action='store_true')
    args = p.parse_args()
    out = args.out.resolve()
    protocol, completion = read(out / 'protocol.json'), read(out / 'completion.json')
    rows = [json.loads(l) for l in (out / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in rows:
        # Counters include each logical request, JSON reask and transport retry.
        r['api_call_count'] = sum(r.get('request_counts', {}).values())
    assert len(rows) == 59 and len({r['id'] for r in rows}) == 59
    assert all(r['id'] != 'rag-biz-007' for r in rows)
    fixture = ROOT / 'tests/rag/fixtures'
    datasets = {'main': load_gold_dataset(fixture / 'rag_gold_v4_review.json', reviewed_only=False),
                'regression': load_gold_dataset(fixture / 'rag_acceptance_v1_review.json', reviewed_only=False)}
    cases = {c.case_id: c for d in datasets.values() for c in d.cases}
    cohorts = {n: sorted([r for r in rows if r['cohort'] == n], key=lambda r: r['id'])
               for n in ('formal48', 'pending024', 'regression10_pending')}
    assert len(cohorts['formal48']) == 48 and all(cases[r['id']].review_status == 'reviewed' for r in cohorts['formal48'])
    assert len(cohorts['pending024']) == 1 and len(cohorts['regression10_pending']) == 10
    groups = {**cohorts, 'original18_eligible17': [r for r in cohorts['formal48'] if int(r['id'][-3:]) <= 18],
              'added32_eligible31': [r for r in cohorts['formal48'] if int(r['id'][-3:]) > 18]}
    summaries = {n: summary(rs) for n, rs in groups.items()}
    issues = []
    for r in sorted(rows, key=lambda r: (r['cohort'], r['id'])):
        c = cases[r['id']]
        j = r.get('gold_judgement') or {}
        findings = [x for x in j.get('rubrics', []) if x['status'] in ('missing', 'contradicted', 'present', 'uncertain') and x['role'] != 'context']
        if j.get('verdict') != 'consistent' or r.get('error'):
            issues.append({'id': r['id'], 'cohort': r['cohort'], 'formal_scored': r['formal_scored'],
                'question': c.question, 'expected_behavior': c.expected_behavior,
                'verdict': j.get('verdict', 'not_judged'), 'behavior_reason': j.get('behavior_reason'),
                'answer': r.get('answer', {}), 'findings': findings, 'error': r.get('error'),
                'context_metrics': r.get('stages', {}).get('context'),
                'reason_limit': 'This is a model-judge finding, requiring review. Missing a declared quote does not by itself prove the cause of a missing answer fact.'})
        r['answer_findings'] = findings
    # Compare only local stages to the immediately preceding run, with identical Gold.
    before_dir = ROOT / 'reports/rag_current_revalidation_20261103'
    before_protocol = read(before_dir / 'protocol.json')
    assert protocol['bundle_id'] == before_protocol['bundle_id']
    assert all(protocol['production_code_sha256'][k] == h for k, h in before_protocol['production_code_sha256'].items())
    for path, digest in protocol['input_sha256'].items():
        assert digest == before_protocol['input_sha256'][path]
    before = {r['id']: r for r in map(json.loads, (before_dir / 'all_cases.jsonl').read_text(encoding='utf-8').splitlines())}
    comparisons = []
    for r in rows:
        if not r.get('stages'):
            continue
        b = before[r['id']]
        assert b['question'] == r['question']
        comparisons.append({'id': r['id'], 'cohort': r['cohort'],
            'context_order_identical': [x['chunk_id'] for x in r.get('candidate_stages', {}).get('context', [])] == [x['chunk_id'] for x in b['candidate_stages'].get('context', [])],
            'context_gold_recall_before': b['stages']['context']['gold_evidence_recall'],
            'context_gold_recall_after': r['stages']['context']['gold_evidence_recall'],
            'specified_negatives_before': b['stages']['context']['hard_negative_ids'],
            'specified_negatives_after': r['stages']['context']['hard_negative_ids']})
    verdict_names = {'consistent': '符合金标', 'incomplete': '遗漏必答事实', 'incorrect': '错误主张',
        'needs_clarification': '没有先澄清', 'safety_gap': '遗漏适用安全要求', 'behavior_mismatch': '回答行为不符',
        'uncertain': '裁判不确定', 'unavailable': '判分未完成', 'not_judged': '未判分'}
    tests = read(out / 'code_regression.json')
    notes = read(out / 'annotation_review_notes.json') if (out / 'annotation_review_notes.json').exists() else []
    result = {'bundle_id': protocol['bundle_id'], 'llm_model': protocol['llm_model'],
        'summaries': summaries, 'completion': completion, 'code_regression': tests,
        'formal_answer_human_accuracy': None,
        'note': '模型裁判符合率是辅助审核指标，不是人工确认的真实准确率。pending诊断不合并进正式分母。',
        'local_stage_comparison': comparisons,
        'local_context_regressions': [x['id'] for x in comparisons if x['context_gold_recall_after'] is not None and x['context_gold_recall_after'] < x['context_gold_recall_before']],
        'local_context_order_changed': [x['id'] for x in comparisons if not x['context_order_identical']],
        'formal_problem_ids': [r['id'] for r in issues if r['formal_scored']],
        'diagnostic_problem_ids': [r['id'] for r in issues if not r['formal_scored']],
        'manual_review_notes': notes}
    write(out / 'summary.json', result)
    write(out / 'answer_review_checklist.json', issues)
    write(out / 'comparison.json', comparisons)
    native_links = {}
    for name, rs in cohorts.items():
        directory = out / name
        directory.mkdir(exist_ok=True)
        metadata = {'run_id': out.name + '-' + name, 'mode': 'full', 'ai_bundle_id': protocol['bundle_id'],
            'started_at': protocol['started_at'], 'finished_at': completion['finished_at'],
            'state': 'completed' if all(r['execution_status'] == 'completed' for r in rs) else 'partial',
            'settings': {'llm_model': protocol['llm_model'], 'semantic_clarification_enabled': protocol['settings']['rag_semantic_clarification_enabled'],
                         'answer_review_enabled': protocol['settings']['rag_answer_review_enabled']},
            'diagnostic_only': name != 'formal48', 'pending_statuses_unchanged': True}
        write(directory / 'run.json', metadata)
        (directory / 'cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, default=str) + '\n' for r in rs), encoding='utf-8')
        dataset_path = datasets['regression' if name == 'regression10_pending' else 'main'].path
        if not args.skip_native:
            generate_report(dataset_path, directory, directory)
        native_links[name] = (directory / 'report.html').relative_to(out).as_posix()
    md = ['# RAG全量问答重新验证报告（2026-11-03）', '',
        f'本轮使用当前模型`{protocol["llm_model"]}`重新运行真实检索、重排、上下文构建、生成、引用检查和独立Gold语义判分。未复用历史答案、历史裁判或重排分数。执行状态：{completion["states"]}。', '',
        '## 范围与配置', '',
        '主集50题：48 reviewed正式统计，007 disputed排除，024 pending只作诊断；回归集10题全部pending，仅诊断。原18题中17题可计、新增32题中31题可计，两组分别报告。', '',
        f'AI bundle_id：`{protocol["bundle_id"]}`；11,769个内容块。源测试集、程序代码和应用配置保持不变。本轮显式使用新版v4评测，不切换应用默认数据集。', '',
        f'语义澄清候选：`{protocol["settings"]["rag_semantic_clarification_enabled"]}`；答案二次复核：`{protocol["settings"]["rag_answer_review_enabled"]}`。按当前实际配置验证，不将关闭功能的结果当作该候选方案验收结论。', '',
        '生成模型只收到问题和检索上下文；金标仅提供给独立判分步骤。同一模型负责生成和裁判，判分属于辅助审核，不能替代业务人员确认。未按结果好坏重跑或修改Gold。', '',
        '## 回答结果', '',
        '| 集合 | 已回答 | 已判分 | 裁判认为符合金标 | 符合比例 |', '|---|---:|---:|---:|---:|']
    for name, v in summaries.items():
        md.append(f'| {GROUP_LABELS[name]} | {v["answered"]}/{v["planned"]} | {v["judged"]}/{v["planned"]} | {v["model_judge_consistent_count"]}/{v["judged"]} | {pct(v["model_judge_consistent_rate"])} |')
    md += ['', '**比例表示模型裁判符合率，不是人工准确率。** 答案是否完整、行为是否正确、证据是否命中分别核对，不将pending回归结果合并成正式分数。', '',
        '原始评测器的行为识别依赖固定短语，可能把自然语言拒答当作回答；本报告行为表使用独立语义裁判，原始启发式字段仅供诊断，不作为最终行为符合率。', '',
        '| 主集预期行为 | 题数 | 行为符合 | 全部检查项符合 |', '|---|---:|---:|---:|']
    for b, s in summaries['formal48']['by_expected_behavior'].items():
        label = {'answer': '直接回答', 'partial_answer': '部分回答', 'clarify': '先澄清', 'refuse': '拒答'}[b]
        md.append(f'| {label} | {s["judged"]} | {s["behavior_match"]} | {s["consistent"]} |')
    md += ['',
        '| 集合 | 判分类别 | 题数 |', '|---|---|---:|']
    for name in ('formal48', 'regression10_pending', 'pending024'):
        for verdict, count in summaries[name]['verdicts'].items():
            md.append(f'| {GROUP_LABELS[name]} | {verdict_names.get(verdict,verdict)} | {count} |')
    md += ['', '## 各阶段证据表现', '', '| 集合 | 阶段 | 至少一条完整引文命中 | 平均证据覆盖 | 指定反例次数 |', '|---|---|---:|---:|---:|']
    for name in ('formal48', 'regression10_pending', 'pending024'):
        for stage in STAGE_ORDER:
            s = summaries[name]['stages'][stage]
            md.append(f'| {GROUP_LABELS[name]} | {STAGE_LABELS[stage]} | {s["gold_hit_cases"]}/{s["cases_with_gold"]} | {pct(s["mean_evidence_recall"])} | {s["hard_negative_candidate_count"]} |')
    md += ['', '拒答题没有正向Gold，不进入引文命中率分母。完整引文覆盖不等于必答事实覆盖；其他有效来源尚未标为等价Gold时，可能出现指定引文未命中但回答仍正确的情况。指定反例出现也不自动等于答案采用了错误结论。', '',
        '## 逐题审核清单', '', '| 集合 | 题号及问题 | 裁判发现 |', '|---|---|---|']
    for r in issues:
        detail = '；'.join(f'{x["id"]} {x["status"]}：{x["gold_text"]}（{x["reason"]}）' for x in r['findings']) or r.get('behavior_reason') or str(r.get('error') or '')
        cells = [GROUP_LABELS[r['cohort']], r['id'] + ' ' + r['question'], verdict_names.get(r['verdict'], r['verdict']) + '：' + detail]
        md.append('| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in cells) + ' |')
    md += ['', '每题实际答案、引文、逐项裁判理由和各阶段候选保存在HTML及answer_review_checklist.json中。裁判结论尚须人工复核，不把所有遗漏都归因于重排。引用校验仅检查S编号和格式，Gold裁判只核对标注的检查项与错误主张，不能据此保证答案新增的所有结论都有充分证据。', '',
        '### 需复核的标注或裁判边界', '']
    for note in notes:
        md.append('- ' + note['id'] + '：' + note['note'] + ' 本轮不更改Gold或分数。')
    md += ['',
        '## 与上一轮的对照', '',
        f'同一份Gold、同一AI包、同一生产代码。相比上一轮本地59题：上下文来源顺序变化题：{result["local_context_order_changed"]}；证据覆盖下降题：{result["local_context_regressions"]}。', '',
        '上一轮被403阻断，没有全量生成和判分结果，因此本轮不能宣称答案准确率相对上一轮提高。现在补齐的是此前缺失的生成与判分证据。', '',
        '## 代码回归与请求记录', '',
        f'`{tests["command"]}`：{tests["result"]}。跳过的真实模型集成测试要求显式环境开关；本轮业务批量运行已实际加载模型。', '',
        '| 集合 | 平均答复耗时（不含判分） | 平均独立判分耗时 | 实际请求尝试数 |', '|---|---:|---:|---:|']
    for name in ('formal48', 'pending024', 'regression10_pending'):
        s = summaries[name]
        md.append(f'| {GROUP_LABELS[name]} | {s["mean_answer_ms"]/1000:.2f}s | {s["mean_judge_ms"]/1000:.2f}s | {sum(s["request_counts"].values())} |')
    md += ['',
        '本地推理串行、最多3个业务任务交叠远程调用。各题执行时间不含等待本地推理锁的排队时间，排队另记local_queue_wait_ms；并发下逐题耗时之和不等于整批墙钟耗时。实际请求与传输重试分开记录，见各题request_counts及protocol.json。', '',
        '## 未计分与未启用项', '',
        '007不执行、不计分；024与10题回归仍pending。未运行候选澄清开启后的对照和13项边界控制，本轮不声称该候选已验收通过。没有修改业务算法、审核状态或应用配置。', '',
        '## 文件', '',
        '- report.html：可搜索、逐题展开的完整报告。',
        '- summary.json：分集合指标；answer_review_checklist.json：需要人工复查的答案。',
        '- cases.jsonl：59题原始执行、答案、候选、裁判与请求次数。',
        '- protocol.json、completion.json：固定配置、哈希和执行完整性。',
        '- formal48、pending024、regression10_pending：原生详细报告。', '']
    (out / 'report.md').write_text('\n'.join(md), encoding='utf-8')
    esc = html.escape
    h = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RAG全量问答验证</title>',
        '<style>body{font:16px/1.65 system-ui;color:#19283b;max-width:1450px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #d1dce9;padding:10px;vertical-align:top}th{background:#edf4fc}details{border:1px solid #d1dce9;margin:10px 0}summary{padding:12px;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:14px;background:#f4f7fb}.notice{padding:18px;background:#edf6ff;border-left:5px solid #2671bb}.warn{padding:15px;background:#fff2d9}input,select{font:inherit;padding:8px;margin:7px}a{color:#165da8}.muted{color:#53657b;font-size:14px}</style>',
        '<h1>RAG全量问答验证 · 2026-11-03</h1>',
        '<p class="notice">本轮重新执行检索、重排、生成、引用校验与Gold判分。<b>正式48题与待审核11题分别统计；007排除。</b><br>模型裁判仅辅助审核，结果不代表人工确认的真实准确率。</p>',
        '<p><a href="report.md">完整文字报告</a> · <a href="summary.json">指标数据</a> · <a href="answer_review_checklist.json">答案审核清单</a></p>',
        '<p>' + esc('模型：' + protocol['llm_model'] + '；AI包：' + protocol['bundle_id'] + '；语义澄清候选=' + str(protocol['settings']['rag_semantic_clarification_enabled']) + '；答案二次复核=' + str(protocol['settings']['rag_answer_review_enabled'])) + '</p>',
        '<h2>回答结果</h2><table><tr><th>集合</th><th>已回答</th><th>已判分</th><th>符合金标</th><th>裁判符合率</th></tr>']
    for name, v in summaries.items():
        h.append(f'<tr><td>{esc(GROUP_LABELS[name])}</td><td>{v["answered"]}/{v["planned"]}</td><td>{v["judged"]}/{v["planned"]}</td><td>{v["model_judge_consistent_count"]}/{v["judged"]}</td><td>{pct(v["model_judge_consistent_rate"])}</td></tr>')
    h += ['</table><p class="muted">原18题排除007后17题；新增32题排除待确认024后31题。回归10题仍pending，不合入正式分数。</p>',
          '<h2>判分类别</h2><table><tr><th>集合</th><th>类别</th><th>数量</th></tr>']
    for name in ('formal48', 'pending024', 'regression10_pending'):
        for verdict, count in summaries[name]['verdicts'].items():
            h.append('<tr><td>' + esc(GROUP_LABELS[name]) + '</td><td>' + esc(verdict_names.get(verdict, verdict)) + f'</td><td>{count}</td></tr>')
    h += ['</table><h2>各阶段表现</h2><table><tr><th>阶段</th><th>主集命中</th><th>主集平均引文覆盖</th><th>回归命中（诊断）</th><th>回归平均覆盖</th></tr>']
    for stage in STAGE_ORDER:
        a, b = summaries['formal48']['stages'][stage], summaries['regression10_pending']['stages'][stage]
        h.append(f'<tr><td>{STAGE_LABELS[stage]}</td><td>{a["gold_hit_cases"]}/{a["cases_with_gold"]}</td><td>{pct(a["mean_evidence_recall"])}</td><td>{b["gold_hit_cases"]}/{b["cases_with_gold"]}</td><td>{pct(b["mean_evidence_recall"])}</td></tr>')
    h += ['</table><h2>逐题答案与审核</h2><input id="search" placeholder="搜索题号、问题" oninput="filterCases()"><select id="cohort" onchange="filterCases()"><option value="">所有集合</option><option value="formal48">主测试集：48题正式</option><option value="pending024">024：待审核诊断</option><option value="regression10_pending">回归集：10题待审核诊断</option></select><label><input id="onlyissues" type="checkbox" onchange="filterCases()">仅裁判发现问题的题</label>']
    for r in sorted(rows, key=lambda r: (r['cohort'], r['id'])):
        c, j = cases[r['id']], r.get('gold_judgement') or {}
        verdict = j.get('verdict', 'not_judged')
        h.append(f'<details class="case" data-cohort="{r["cohort"]}" data-problem="{int(verdict!="consistent")}"><summary>' + esc(r['id'] + ' · ' + c.question + ' · ' + verdict_names.get(verdict, verdict)) + '</summary>')
        h.append('<p>' + esc(f'集合={GROUP_LABELS[r["cohort"]]}；审核状态={c.review_status}；预期={c.expected_behavior}；程序状态={r.get("answer",{}).get("status","未生成")}；裁判行为匹配={j.get("behavior","未判分")}') + '</p><h3>实际最终答案</h3><pre>' + esc(r.get('answer', {}).get('text', '未获得答案')) + '</pre>')
        for note in notes:
            if note['id'] == r['id']:
                h.append('<p class="warn">待人工复核：' + esc(note['note']) + '（本轮不改标签和分数）</p>')
        if r.get('error'):
            h.append('<p class="warn">' + esc(str(r['error'])) + '</p>')
        h.append('<h3>逐项核对（模型裁判，待人工复查）</h3><p>' + esc(j.get('behavior_reason', '')) + '</p><table><tr><th>检查项</th><th>状态</th><th>裁判理由</th></tr>')
        for x in j.get('rubrics', []):
            h.append('<tr><td>' + esc(x['id'] + ' · ' + x['gold_text']) + '</td><td>' + esc(x['status']) + '</td><td>' + esc(x['reason']) + '</td></tr>')
        h.append('</table><details><summary>Gold原文及来源</summary>')
        for e in c.gold_evidence:
            h.append('<p>' + esc(str(e['file']) + ' · 第' + str(e.get('page', '')) + '页/张') + '</p><pre>' + esc(e['quote']) + '</pre>')
        h.append('</details><details><summary>本次实际上下文</summary>')
        for x in r.get('candidate_stages', {}).get('context', []):
            h.append('<p>' + esc(str(x.get('filename')) + ' · ' + str(x.get('location'))) + '</p><pre>' + esc(x.get('included_text') or x.get('text') or '') + '</pre>')
        h.append('</details><details><summary>阶段指标、请求及耗时</summary><pre>' + esc(json.dumps({k: r.get(k) for k in ('stages', 'request_counts', 'duration_ms', 'local_queue_wait_ms')}, ensure_ascii=False, indent=2)) + '</pre></details></details>')
    h += ['<h2>原生详细报告</h2><ul>']
    for name, link in native_links.items():
        h.append('<li><a href="' + esc(link, quote=True) + '">' + esc(GROUP_LABELS[name]) + '</a></li>')
    h += ['</ul><p>未执行候选澄清开启后的实验和13项远程边界控制，本报告不将默认关闭状态当作候选验收。</p>',
        '<script>function filterCases(){const q=document.querySelector("#search").value.toLowerCase(),c=document.querySelector("#cohort").value,p=document.querySelector("#onlyissues").checked;document.querySelectorAll("details.case").forEach(e=>e.hidden=!(e.querySelector("summary").textContent.toLowerCase().includes(q)&&(!c||e.dataset.cohort===c)&&(!p||e.dataset.problem==="1")))}</script></html>']
    (out / 'report.html').write_text('\n'.join(h), encoding='utf-8')
    print(json.dumps({'formal': summaries['formal48']['verdicts'],
        'regression': summaries['regression10_pending']['verdicts'], 'pending024': summaries['pending024']['verdicts'],
        'problems': result['formal_problem_ids']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
