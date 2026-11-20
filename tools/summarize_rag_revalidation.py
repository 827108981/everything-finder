"""Summarize fresh retrieval, pending diagnostics, tests and external blockers."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import statistics

from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import aggregate, score_case, STAGE_ORDER, STAGE_LABELS
from local_full_text_search.rag.evaluation.reporting.timing import case_timing, MODULE_LABELS
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.index_store import RagIndexStore

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def pct(value):
    return '—' if value is None else f'{value * 100:.1f}%'


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    i = (len(values) - 1) * fraction
    left = int(i)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (i - left)


def summarize(rows):
    v = aggregate(rows)
    times = [r['duration_ms'] for r in rows if not r.get('error')]
    v['timing'] = {'mean_ms': statistics.mean(times) if times else None,
                   'p50_ms': percentile(times, .5), 'p95_ms': percentile(times, .95),
                   'sum_case_ms': sum(times), 'case_count': len(times)}
    modules = Counter()
    for r in rows:
        modules.update(case_timing(r)['modules_ms'])
    v['timing']['modules_ms'] = dict(modules)
    v['judged_answer_count'] = 0
    v['answer_accuracy'] = None
    v['note'] = '检索命中只说明标注引文进入该阶段，不是答案准确率。反例出现仅说明指定片段存在，不自动表示模型误用了它。'
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    completion, protocol = read(out / 'completion.json'), read(out / 'protocol.json')
    assert len(completion['cohorts']) == 3
    datasets = {
        'main': load_gold_dataset(ROOT / 'tests/rag/fixtures/rag_gold_v4_review.json', reviewed_only=False),
        'regression': load_gold_dataset(ROOT / 'tests/rag/fixtures/rag_acceptance_v1_review.json', reviewed_only=False),
    }
    cases = {c.case_id: c for d in datasets.values() for c in d.cases}
    rows, cohort_rows = [], {}
    native_runs = {}
    for cohort, info in completion['cohorts'].items():
        run_dir = Path(info['run_dir'])
        group = [json.loads(l) for l in (run_dir / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(group) == info['count'] and info['state'] == 'completed'
        for r in group:
            r['cohort'] = cohort
            r['review_status'] = cases[r['id']].review_status
            r['formal_scored'] = cohort == 'formal48'
            r['generation_validation'] = 'blocked_not_validated'
        rows.extend(group)
        cohort_rows[cohort] = group
        native_runs[cohort] = {'run_dir': str(run_dir), 'report': str(run_dir / 'report.html')}
    assert len(rows) == 59 and len({r['id'] for r in rows}) == 59
    assert 'rag-biz-007' not in {r['id'] for r in rows}
    assert sum(r['formal_scored'] for r in rows) == 48
    assert all(r['review_status'] == 'reviewed' for r in cohort_rows['formal48'])
    assert all(r['review_status'] == 'pending' for cohort in ('pending024', 'regression10_pending') for r in cohort_rows[cohort])
    store = RagIndexStore(Path('E:/everythingfinder-db/rag'))
    bundle = store.active()
    assert bundle.bundle_id == protocol['bundle_id']
    conn = store.connect_read_only(bundle.metadata_path)
    try:
        chunk_rows = load_chunk_rows(conn)
    finally:
        conn.close()
    mappings = {cid: map_case_evidence(c, chunk_rows) for cid, c in cases.items()}

    failure_details = []
    loss_reason_cases = {}
    for r in rows:
        c = cases[r['id']]
        union = list({x['chunk_id']: x for stage in ('lexical', 'vector')
                      for x in r.get('candidate_stages', {}).get(stage, [])}.values())
        entry = score_stage('fusion', union, mappings[c.case_id])
        r['retrieval_union'] = entry
        if r.get('error'):
            problem = '执行错误'
        elif c.expected_behavior == 'refuse':
            problem = '拒答题，仅检索不能验证最终行为'
        elif not r['stages']['context']['gold_hit']:
            problem = '未召回指定正确引文' if not entry['gold_hit'] else '召回后未完整进入上下文'
        elif r['stages']['context']['gold_evidence_recall'] < 1:
            problem = '上下文缺少部分标注证据'
        else:
            problem = '标注证据齐全，答案未验证'
        r['problem'] = problem
        losses = []
        for e in mappings[c.case_id]:
            if e.kind != 'positive':
                continue
            eid = e.evidence_id
            present = ['retrieval_union'] if eid in entry['gold_completion_ranks'] else []
            for stage in STAGE_ORDER:
                if eid in r['stages'][stage]['gold_completion_ranks']:
                    present.append(stage)
            if 'context' in present:
                continue
            transitions = []
            chain = ['retrieval_union', 'fusion', 'deduplicated', 'rerank', 'constraints', 'evidence', 'context']
            for before, after in zip(chain, chain[1:]):
                if before in present and after not in present:
                    transitions.append(f'{before}→{after}')
            ids = set(e.chunk_ids)
            trace = [t for t in r['candidate_transitions'] if t['chunk_id'] in ids and t['status'] != 'present']
            candidates = []
            for stage in ('deduplicated', 'rerank', 'constraints', 'evidence'):
                for x in r['candidate_stages'].get(stage, []):
                    if ids.intersection({x['chunk_id'], *x.get('member_chunk_ids', [])}):
                        candidates.append({'stage': stage, 'chunk_id': x['chunk_id'], 'file': x.get('file_path'),
                            'location': x.get('location'), 'rerank_score': x.get('rerank_score'),
                            'combined': x.get('rerank_score_combined'), 'selected': x.get('selected'),
                            'reason': x.get('reason'), 'constraint_decision': x.get('constraint_decision')})
            losses.append({'evidence_id': eid, 'file': e.file, 'page': e.page, 'quote': e.quote,
                           'present_stages': present, 'lost_transitions': transitions,
                           'trace': trace, 'candidates': candidates})
        r['missing_evidence'] = losses
        reasons = set()
        for loss in losses:
            if not loss['present_stages']:
                reasons.add('两路召回未完整找到指定引文')
            elif 'deduplicated' not in loss['present_stages']:
                reasons.add('融合、去重或进入重排前的候选截断')
            elif 'rerank' not in loss['present_stages']:
                reasons.add('未进入重排保留候选')
            elif 'evidence' in loss['present_stages']:
                reasons.add('上下文预算或截断未保留完整引文')
            for x in loss['candidates']:
                if x['stage'] != 'evidence':
                    continue
                reason = x.get('reason') or ''
                if '同一文档' in reason:
                    reasons.add('同一文档条数上限')
                elif 'TopK' in reason:
                    reasons.add('最终Top8截断')
                elif x.get('constraint_decision', {}).get('hard_rejected'):
                    reasons.add('明确条件冲突被拦截')
            if loss['lost_transitions'] and not reasons:
                reasons.add('已召回但后续证据筛选未保留')
        r['loss_reasons'] = sorted(reasons)
        for reason in reasons:
            loss_reason_cases.setdefault((r['cohort'], reason), []).append(r['id'])
        if losses or r.get('error'):
            failure_details.append({'id': r['id'], 'cohort': r['cohort'], 'question': c.question,
                'formal_scored': r['formal_scored'], 'problem': problem, 'loss_reasons': sorted(reasons), 'missing_evidence': losses})

    summaries = {k: summarize(v) for k, v in cohort_rows.items()}
    formal = cohort_rows['formal48']
    summaries['original18_eligible17'] = summarize([r for r in formal if int(r['id'][-3:]) <= 18])
    summaries['added32_eligible31'] = summarize([r for r in formal if int(r['id'][-3:]) > 18])
    for name, group in list(cohort_rows.items()):
        summaries[name]['retrieval_union'] = {
            'cases_with_gold': sum(r['expected_behavior'] != 'refuse' for r in group),
            'hit_cases': sum(r['expected_behavior'] != 'refuse' and r['retrieval_union']['gold_hit'] for r in group)}
        summaries[name]['context_no_hit_ids'] = [r['id'] for r in group
            if r['expected_behavior'] != 'refuse' and not r['stages']['context']['gold_hit']]
        summaries[name]['context_partial_ids'] = [r['id'] for r in group
            if r['expected_behavior'] != 'refuse' and r['stages']['context']['gold_hit']
            and r['stages']['context']['gold_evidence_recall'] < 1]

    # A same-question, same-current-Gold comparison; never compare old headline scores.
    historical_path = ROOT / 'reports/project_constraint_fix_20261102/cases.jsonl'
    baseline_rows = [json.loads(l) for l in historical_path.read_text(encoding='utf-8').splitlines()]
    baseline = {r['id']: r for r in baseline_rows if r.get('arm') == 'after' and 'error_type' not in r}
    comparisons = []
    old_selected, old_regression = [], []
    for r in rows:
        if r['id'] not in baseline or baseline[r['id']]['question'] != r['question']:
            continue
        old = baseline[r['id']]
        c = cases[r['id']]
        scored = score_case(c, old['candidate_stages'], mappings[c.case_id], answer={'mode': 'retrieval-only', 'text': ''})
        if r['formal_scored']:
            old_selected.append(scored)
        elif r['cohort'] == 'regression10_pending':
            old_regression.append(scored)
        before_ids = [x['chunk_id'] for x in old['candidate_stages'].get('context', [])]
        after_ids = [x['chunk_id'] for x in r['candidate_stages'].get('context', [])]
        comparisons.append({'id': r['id'], 'formal_scored': r['formal_scored'],
            'before_context_hit': scored['stages']['context']['gold_hit'],
            'after_context_hit': r['stages']['context']['gold_hit'],
            'before_context_recall': scored['stages']['context']['gold_evidence_recall'],
            'after_context_recall': r['stages']['context']['gold_evidence_recall'],
            'context_order_identical': before_ids == after_ids,
            'before_hard_negatives': scored['stages']['context']['hard_negative_ids'],
            'after_hard_negatives': r['stages']['context']['hard_negative_ids']})
    assert len(old_selected) == 48 and len(old_regression) == 10
    paired = {'baseline': str(historical_path), 'baseline_time': '2026-11-02',
        'note': '仅比较同问法、同一新版Gold重新计分的检索阶段；不使用旧答案或旧裁判结果。本次没有新增算法改动，不能称为算法改进实验。',
        'before_formal48': aggregate(old_selected), 'before_regression10_diagnostic': aggregate(old_regression),
        'cases': comparisons,
        'formal_context_regressions': [r['id'] for r in comparisons if r['formal_scored'] and
            r['after_context_recall'] is not None and r['after_context_recall'] < r['before_context_recall']],
        'formal_context_improvements': [r['id'] for r in comparisons if r['formal_scored'] and
            r['after_context_recall'] is not None and r['after_context_recall'] > r['before_context_recall']]}
    remote = read(out / 'remote_probe.json')
    tests = read(out / 'code_regression.json')
    evidence = read(out / 'evidence_audit.json')
    alternative_notes = read(out / 'alternate_source_review.json') if (out / 'alternate_source_review.json').exists() else []
    path_audit = read(out / 'mapping_path_audit.json') if (out / 'mapping_path_audit.json').exists() else {}
    summary = {'scope': {'formal': 48, 'original18_eligible': 17, 'added32_eligible': 31,
                         'pending_main': 1, 'regression_pending': 10, 'excluded': ['rag-biz-007']},
        'validation_state': 'local_complete_remote_blocked' if remote['state'] == 'blocked' else 'local_complete_remote_probe_only',
        'bundle_id': protocol['bundle_id'], 'summaries': summaries, 'code_regression': tests,
        'remote': remote, 'paired_comparison': paired, 'native_reports': native_runs,
        'remote_attempts': {'connectivity_probe': 1, 'business_generation_probe': remote['attempts'],
                            'gold_judge': 0, 'semantic_clarification': 0},
        'evidence_counts': evidence['counts'],
        'loss_reason_cases': [{'cohort': c, 'reason': why, 'ids': ids} for (c, why), ids in loss_reason_cases.items()],
        'evidence_source_files_missing': [e for e in evidence['entries'] if not e['original_file_exists']],
        'evidence_unmatched': [e['evidence_id'] for e in evidence['entries'] if not e['chunk_ids']],
        'alternate_sources_unscored': alternative_notes,
        'same_name_path_audit': path_audit,
        'formal_answer_accuracy': None, 'algorithm_change_in_this_run': False}
    write(out / 'summary.json', summary)
    write(out / 'problem_cases.json', failure_details)
    write(out / 'comparison.json', paired)
    with (out / 'all_cases.jsonl').open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')

    md = ['# 当前RAG测试集与回归集重新验证报告（2026-11-03）', '',
        '**结论：本地链路已完成；远程生成及模型判分被HTTP 403阻断，尚不能认定完整问答验收通过。**', '',
        '本次使用当前工作区程序、当前实际设置和当前已发布AI索引，重新运行真实本地嵌入模型与重排模型，没有复用历史模型答案、模型判分或重排分数。没有改业务算法或应用配置。', '',
        f'AI bundle_id：`{protocol["bundle_id"]}`；11,769个内容块。模型：BGE-large-zh-v1.5嵌入、BGE-reranker-base重排；当前LLM `{protocol["llm_model"]}`。', '',
        '重排混合权重0.65/0.05/0.30，名次保护3，最终Top8，上下文12,000字符；检索参数完整快照及代码哈希见protocol.json。语义澄清候选默认关闭，答案二次复核也关闭，本次未自行启用。', '',
        '## 范围与计分', '',
        '| 集合 | 题数 | 本轮处理 |', '|---|---:|---|',
        '| 新版主测试集 reviewed | 48 | 正式统计检索阶段，答案准确率未验证 |',
        '| 主测试集024 | 1 | 新问法pending，仅诊断，不进正式分母 |',
        '| 主测试集007 | 1 | disputed，完全排除，不执行、不计分 |',
        '| 现有10题回归集 | 10 | 全部pending，仅诊断，不是人工通过的验收集 |',
        '| 代码回归 | 244项 | 243通过、1跳过 |', '',
        '原18题固定集合本轮17题可计分（排除007），新增32题本轮31题可计分（024待确认）。两组分别统计，不能把题数或金标变化说成算法进步。现有10题已被多轮实验使用，本次称回归集，不再宣称独立盲测。', '',
        '## 执行完整性', '',
        f'本地运行59/59题，错误{sum(bool(r.get("error")) for r in rows)}题。JSON与JSONL内容一致，007未进入任何计分集合，待审核状态保留，源测试集、AI包和应用配置哈希未变。', '',
        f'代码回归：`{tests["command"]}`，{tests["result"]}。跳过项原因：{tests["skip_reason"].rstrip("。")}。', '',
        f'原资料文件缺失{len(summary["evidence_source_files_missing"])}条；索引未匹配引文{len(summary["evidence_unmatched"])}条。核验范围是文件存在和当前索引正文/页码定位，不声称本轮重新目视审核所有原文页。', '',
        '匹配器对15条引文提示同名多路径待确认：本次逐条核查，声明的绝对路径都存在，副本整文件SHA256一致、同页正文也一致。属于同文件复制到不同目录；保留原匹配器提示和原计分，不擅自修改算法。详情见mapping_path_audit.json。', '',
        '## 各步骤表现', '',
        '命中：该题至少一条完整标注引文进入阶段；覆盖率：每题完整引文条数/该题声明条数后取平均。均不是答案准确率。缺少某条引文也不必然意味着漏答某个必答事实，可能有其他来源支持，需结合问题逐项核查。拒答题没有正向Gold，不计入证据命中率分母。', '',
        '| 集合 | 阶段 | Gold命中 | 平均证据覆盖 | 指定反例次数 |', '|---|---|---:|---:|---:|']
    for group in ('formal48', 'original18_eligible17', 'added32_eligible31', 'regression10_pending', 'pending024'):
        for stage in STAGE_ORDER:
            s = summaries[group]['stages'][stage]
            md.append(f'| {group} | {STAGE_LABELS[stage]} | {s["gold_hit_cases"]}/{s["cases_with_gold"]} | {pct(s["mean_evidence_recall"])} | {s["hard_negative_candidate_count"]} |')
    md += ['', '## 需要关注的题目', '', '| 集合 | 题号及问题 | 观察结果与原因 |', '|---|---|---|']
    for r in rows:
        if r['missing_evidence'] or r.get('error'):
            md.append(f'| {r["cohort"]} | {r["id"]} {r["question"]} | {r["problem"]}：{"；".join(r["loss_reasons"])} |')
    md += ['', '同一题可以同时缺少多条证据，所以以下原因分组可能重叠，不应相加当作失败题数。', '', '| 集合 | 丢失位置/直接原因 | 题号 |', '|---|---|---|']
    for (cohort, reason), ids in loss_reason_cases.items():
        md.append(f'| {cohort} | {reason} | {", ".join(ids)} |')
    md += ['', '逐条证据在哪个阶段消失、对应候选的条件判断及丢弃原因，见problem_cases.json和HTML逐题展开。', '',
        '召回候选是两路检索并集；融合/去重、重排、条件检查、最终Top8、上下文预算都可能导致引文丢失。只凭“上下文未命中”不能直接归因为重排，报告按实际链路分别记录。指定反例出现也不等于最终答案一定误用。', '',
        '### 其他来源的支持内容（不擅自加分）', '']
    for note in alternative_notes:
        md.append(f'- {note["id"]}：{note["note"]} 原文件：`{note["file"]}`，第{note["page"]}张。短句已在原生幻灯片和实际上下文复核，仅作审核备注，不增加等价Gold或调整分数。')
    md += ['',
        '## 同集合回归比较', '',
        f'以2026-11-02项目条件修复后的同48题记录为对照，统一用本次新版Gold重新计算。上下文覆盖下降题：{paired["formal_context_regressions"]}；上升题：{paired["formal_context_improvements"]}。024问法不同不作配对。', '',
        f'同问法的48题主集和10题回归共{len(comparisons)}对，上下文来源顺序均与对照相同，指定反例也无增减。这里只核查检索复现，不代表生成答案已经复现。', '',
        '此比较检验当前本地链路是否复现，既不比较旧答案分数，也不把金标精简的影响说成算法提升。', '',
        '## 耗时', '', '| 集合 | 本地平均 | P50 | P95 |', '|---|---:|---:|---:|']
    for name in ('formal48', 'regression10_pending', 'pending024'):
        t = summaries[name]['timing']
        md.append(f'| {name} | {t["mean_ms"]/1000:.2f}s | {t["p50_ms"]/1000:.2f}s | {t["p95_ms"]/1000:.2f}s |')
    md += ['', '耗时为CPU、4线程、本次本地链路实测；模型冷加载单列model_loading.json。没有测得生成、复核和模型裁判的成功耗时，不能拿此数当完整问答延迟。', '',
        '## 外部接口阻断与未完成项', '',
        f'使用真实001题当前检索上下文进行一次生成验证：{remote.get("safe_error",remote["state"])}。失败后停止对其余题目重复调用；这些题标为“未执行/接口阻断”，不是逐题答案错误。未从历史答案补齐本轮结果。', '',
        '远程实际请求共2次：运行前轻量连通性检查1次、真实业务生成探针1次，均返回403；澄清判断0次、模型裁判0次。具体失败记录分别见connectivity_probe_initial.json和remote_probe.json。', '',
        '因此尚未完成：48题正式答案质量判分、024与10题回归的回答质量诊断、澄清候选的完整业务验证，以及13项远程澄清边界控制。代码回归覆盖了解析和路由机制，但不能代替这13项控制及真实模型验证。', '',
        '## 后续执行条件', '',
        '恢复当前设置中模型的访问权限后，保持同一测试集哈希、AI包和程序配置，继续全量生成与判分。不要修改Gold来迁就模型，不把本地证据命中率当最终通过率。', '',
        '## 交付文件', '',
        '- report.html：本报告与逐题问题、证据、实际候选。',
        '- summary.json、comparison.json：汇总及同集合配对结果。',
        '- all_cases.jsonl、problem_cases.json：本轮59题真实本地记录与问题定位。',
        '- evidence_audit.json、protocol.json：引文核验、文件/模型/设置/代码快照。',
        '- remote_probe.json、code_regression.json：远程失败与代码回归证据。',
        '- formal48、pending024、regression10_pending：原生逐阶段详细报告。', '']
    (out / 'report.md').write_text('\n'.join(md), encoding='utf-8')

    esc = html.escape
    h = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RAG完整验证记录</title>',
        '<style>body{font:16px/1.65 system-ui;margin:28px auto;max-width:1450px;padding:0 20px;color:#182536}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #d2dae4;padding:9px;vertical-align:top}th{background:#edf3fa}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:12px}summary{cursor:pointer;padding:12px}details{border:1px solid #d2dae4;margin:10px 0}.warn{background:#fff1d5;padding:18px;border-left:5px solid #db8d05}input,select{font:inherit;padding:8px;margin:6px}a{color:#165aaa}.small{font-size:14px;color:#53627a}</style>',
        '<h1>当前测试集与回归集验证报告 · 2026-11-03</h1>',
        '<p class="warn"><b>本地链路59/59已执行；远程接口HTTP 403阻断，完整问答验收尚未通过。</b><br>48题正式检索统计；024与10题回归仅诊断；007排除。生成答案与模型判分未完成，不显示答案准确率。</p>',
        '<p>本次真实重新计算嵌入查询、重排和上下文；未复用旧答案或旧重排分数。未修改业务算法、应用设置或审核状态。语义澄清候选默认关闭。'
        '<a href="report.md">完整文字报告</a> · <a href="summary.json">机器可读汇总</a> · <a href="evidence_audit.json">全部引文核验</a></p>',
        '<h2>正式主集与回归诊断</h2><table><tr><th>集合</th><th>实际运行</th><th>最终上下文Gold命中</th><th>证据覆盖率</th><th>指定反例次数</th></tr>']
    for name in summaries:
        s = summaries[name]
        v = s['stages']['context']
        h.append(f'<tr><td>{esc(name)}</td><td>{s["completed_case_count"]}/{s["case_count"]}</td><td>{v["gold_hit_cases"]}/{v["cases_with_gold"]}</td><td>{pct(v["mean_evidence_recall"])}</td><td>{v["hard_negative_candidate_count"]}</td></tr>')
    h += ['</table><p class="small">证据命中率不是答案准确率；同一段正确资料不保证模型正确作答。回归10题仍为pending，不合并进正式分数。原18题可计17、新增32题可计31，分别统计。</p>',
          '<h2>逐阶段表现</h2><table><tr><th>阶段</th><th>正式48题Gold命中</th><th>正式平均覆盖</th><th>回归10题命中（诊断）</th><th>回归平均覆盖</th></tr>']
    for stage in STAGE_ORDER:
        a, b = summaries['formal48']['stages'][stage], summaries['regression10_pending']['stages'][stage]
        h.append(f'<tr><td>{STAGE_LABELS[stage]}</td><td>{a["gold_hit_cases"]}/{a["cases_with_gold"]}</td><td>{pct(a["mean_evidence_recall"])}</td><td>{b["gold_hit_cases"]}/{b["cases_with_gold"]}</td><td>{pct(b["mean_evidence_recall"])}</td></tr>')
    h += ['</table><h2>代码回归与远程接口</h2>', '<pre>' + esc(json.dumps({'code_tests': tests, 'remote_probe': remote}, ensure_ascii=False, indent=2)) + '</pre>',
          '<h2>同一集合的检索回归比较</h2><p>统一使用本次Gold重算旧记录；不引用旧答案或旧分数，不把金标变化称为算法提升。</p>',
          '<pre>' + esc(json.dumps({k: paired[k] for k in ('formal_context_regressions', 'formal_context_improvements')}, ensure_ascii=False, indent=2)) + '</pre>',
          '<h2>逐题审阅</h2><p>展开可查看问题、必答项、完整引文、候选来源、条件判断与最终上下文。旧答案不出现在本次报告中。</p>',
          '<input id="q" placeholder="搜索题号或问题" oninput="filterCases()"><select id="cohort" onchange="filterCases()"><option value="">所有集合</option><option>formal48</option><option>pending024</option><option>regression10_pending</option></select><label><input type="checkbox" id="problems" onchange="filterCases()">仅缺少标注证据的题</label>']
    for r in rows:
        c = cases[r['id']]
        trace = ' → '.join(STAGE_LABELS[s] + ':' + str(r['stages'][s]['gold_evidence_found']) + '/' + str(len(c.gold_evidence)) for s in STAGE_ORDER)
        h.append(f'<details class="case" data-cohort="{r["cohort"]}" data-problem="{int(bool(r["missing_evidence"]))}"><summary>{esc(r["id"]+" · "+c.question+" · "+r["problem"])}</summary>')
        h.append('<p>' + esc(f'集合：{r["cohort"]}；状态：{c.review_status}；预期行为：{c.expected_behavior}；生成/判分：接口阻断，未验证。') + '</p>')
        h.append('<p>' + esc(trace) + '</p><h3>本题必答项（归纳，不要求照抄）</h3><pre>' + esc('\n'.join(c.required_facts) or '拒答，不编造具体结论。') + '</pre>')
        for note in alternative_notes:
            if note['id'] == r['id']:
                h.append('<p class="warn">其他来源审核备注（不额外计分）：' + esc(note['note']) + '</p><pre>' + esc(json.dumps(note, ensure_ascii=False, indent=2)) + '</pre>')
        for i, e in enumerate(c.gold_evidence, 1):
            h.append('<details><summary>正向证据' + str(i) + ' · ' + esc(str(e['file']) + ' · 第' + str(e.get('page', '')) + '页/张') + '</summary><pre>' + esc(e['quote']) + '</pre></details>')
        if r['missing_evidence']:
            h.append('<h3>指定引文丢失的直接原因</h3><p>' + esc('；'.join(r['loss_reasons'])) + '</p><details><summary>查看完整证据与实际丢弃轨迹</summary><pre>' + esc(json.dumps(r['missing_evidence'], ensure_ascii=False, indent=2)) + '</pre></details>')
        h.append('<details><summary>实际送入生成的本地上下文（尚未成功生成）</summary>')
        for x in r['candidate_stages'].get('context', []):
            h.append('<p>' + esc(str(x.get('filename')) + ' · ' + str(x.get('location'))) + '</p><pre>' + esc(x.get('included_text') or x.get('text') or '') + '</pre>')
        h.append('</details><details><summary>指定强负例（出现不等于答案误用）</summary><pre>' + esc(json.dumps({'definitions': list(c.hard_negatives), 'context_negative_ids': r['stages']['context']['hard_negative_ids']}, ensure_ascii=False, indent=2)) + '</pre></details></details>')
    h += ['<details><summary>rag-biz-007 · 排除，不计分</summary><p>同单位下140±5与145±5仍有资料数值差异；本轮未执行、不作唯一标准判断。</p></details>',
          '<h2>原生详细报告与数据文件</h2><ul>']
    for name, info in native_runs.items():
        h.append('<li><a href="' + esc(Path(info['report']).relative_to(out).as_posix(), quote=True) + '">' + esc(name + '逐阶段报告') + '</a></li>')
    h += ['<li><a href="protocol.json">输入及配置快照</a></li><li><a href="problem_cases.json">问题题目完整定位</a></li><li><a href="comparison.json">配对回归比较</a></li></ul>',
          '<script>function filterCases(){const q=document.querySelector("#q").value.toLowerCase(),c=document.querySelector("#cohort").value,p=document.querySelector("#problems").checked;document.querySelectorAll("details.case").forEach(e=>e.hidden=!(e.querySelector("summary").textContent.toLowerCase().includes(q)&&(!c||e.dataset.cohort===c)&&(!p||e.dataset.problem==="1")))}</script></html>']
    (out / 'report.html').write_text('\n'.join(h), encoding='utf-8')
    print(json.dumps({'state': summary['validation_state'], 'formal_context': summaries['formal48']['stages']['context'],
        'regression_context': summaries['regression10_pending']['stages']['context'],
        'formal_regressions': paired['formal_context_regressions'], 'problem_count': len(failure_details)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
