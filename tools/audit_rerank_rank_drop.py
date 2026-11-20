"""Explain saved rerank scores without rerunning retrieval, models or APIs."""
from copy import deepcopy
import base64
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
from types import SimpleNamespace
from zipfile import ZipFile

from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.evaluation.gold_loader import GoldDataset, load_gold_dataset
from local_full_text_search.rag.evaluation.reporting import service

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'reports/rag_support_graph_20261113'
OUT = ROOT / 'reports/rag_rerank_rank_diagnosis_20261116'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    OUT.mkdir(exist_ok=True)
    rows = lines(HISTORY / 'A.jsonl')
    mappings = read(HISTORY / 'native_gold_mappings.json')['mappings']
    manifest = read(HISTORY / 'A_manifest.json')
    production = {r['id']: r for r in lines(ROOT / 'reports/rag_selection_a_adoption_20261113/production_A.jsonl')}
    weights = [manifest['settings'][k] for k in ('rag_rerank_cross_encoder_weight', 'rag_rerank_retrieval_prior_weight', 'rag_rerank_answer_support_weight')]
    guard = object.__new__(LocalBgeReranker)
    guard.max_rank_drop = manifest['settings']['rag_rerank_max_rank_drop']
    audits = []; verified = 0
    for r in rows:
        xs = r['candidate_stages']['rerank']
        pure = sorted(xs, key=lambda x: (-x['rerank_score'], -x.get('fused_score', 0), x['chunk_id']))
        mixed = sorted(xs, key=lambda x: (-x['rerank_score_combined'], -x.get('fused_score', 0), x['chunk_id']))
        objects = [SimpleNamespace(**x) for x in mixed]
        replay = guard._apply_rank_guard(objects)
        assert [x.chunk_id for x in replay] == [x['chunk_id'] for x in xs]
        for x in xs:
            ranks = [x[k] for k in ('lexical_rank', 'vector_rank', 'fused_rank') if x.get(k)]
            prior = sum(1 / rank for rank in ranks) / len(ranks) if ranks else 0
            x['_prior'] = prior
            x['_components'] = [weights[0] * x['rerank_score'], weights[1] * prior, weights[2] * x['answer_support_score']]
            assert abs(sum(x['_components']) - x['rerank_score_combined']) < 1e-9
            verified += 1
        # Reproduce every guard decision, including lower-score candidates forced forward.
        remaining = list(mixed); steps = []
        for pos in range(1, len(xs) + 1):
            due = [x for x in remaining if x.get('fused_rank') is not None and x['fused_rank'] + guard.max_rank_drop <= pos]
            pool = due or [x for x in remaining if x.get('fused_rank') is None or x['fused_rank'] + guard.max_rank_drop >= pos] or remaining
            selected = pool[0]; remaining.remove(selected)
            steps.append({'position': pos, 'forced_by_guard': bool(due), 'chunk_id': selected['chunk_id'], 'file': selected['filename'], 'location': selected['location'], 'fusion_rank': selected.get('fused_rank'), 'combined_score': selected['rerank_score_combined']})
        ids = {cid for e in mappings.get(r['id'], []) if e.get('kind') == 'positive' for cid in e.get('chunk_ids', [])}
        def hit(x):
            return bool(ids & {x['chunk_id'], *x.get('member_chunk_ids', [])})
        dedup = r['candidate_stages']['deduplicated']
        old = [i for i, x in enumerate(dedup, 1) if hit(x)]
        new = [i for i, x in enumerate(xs, 1) if hit(x)]
        if not r['id'].startswith('rag-biz-') or not old or not new or min(new) <= min(old):
            continue
        candidates = []
        for i, x in enumerate(xs, 1):
            candidates.append({**x, 'positive_mapping': hit(x), 'rank_dedup': next((j for j, y in enumerate(dedup, 1) if y['chunk_id'] == x['chunk_id']), None), 'rank_model': next(j for j, y in enumerate(pure, 1) if y['chunk_id'] == x['chunk_id']), 'rank_combined': next(j for j, y in enumerate(mixed, 1) if y['chunk_id'] == x['chunk_id']), 'rank_final': i})
        entry = {'id': r['id'], 'question': r['question'], 'review_status': r['review_status'], 'dedup_positive_ranks': old, 'model_positive_ranks': [i for i, x in enumerate(pure, 1) if hit(x)], 'combined_positive_ranks': [i for i, x in enumerate(mixed, 1) if hit(x)], 'final_positive_ranks': new, 'candidates': candidates, 'guard_steps': steps, 'missing_before_rerank': [x['chunk_id'] for x in dedup if hit(x) and x['chunk_id'] not in {y['chunk_id'] for y in xs}]}
        audits.append(entry)
    code_checks = {}
    for relative in ('reranking/local_bge.py', 'reranking/support.py'):
        current = hashlib.sha256((ROOT / 'local_full_text_search/rag' / relative).read_bytes()).hexdigest()
        saved = [v for k, v in manifest['code_hashes'].items() if k.replace('\\', '/').endswith(relative)]
        assert saved == [current]
        code_checks[relative] = current
    write(OUT / 'diagnosis.json', {'bundle': manifest['bundle'], 'input': str(HISTORY / 'A.jsonl'), 'input_sha256': hashlib.sha256((HISTORY / 'A.jsonl').read_bytes()).hexdigest(), 'weights': weights, 'max_rank_drop': guard.max_rank_drop, 'formula_verified_candidates': verified, 'guard_reproduced_cases': len(rows), 'code_hashes_verified': code_checks, 'api_calls': 0, 'model_runs': 0, 'scope': 'Main-suite cases whose first mapped-positive rank decreased; other positives shown individually. A rank is not evidence completeness or answer correctness.', 'cases': audits})
    table = '|题号|去重|仅模型|加辅助分|加排名保护（报告重排）|状态|\n|---|---:|---:|---:|---:|---|\n'
    for a in audits:
        vals = [','.join(map(str, a[k])) for k in ('dedup_positive_ranks', 'model_positive_ranks', 'combined_positive_ranks', 'final_positive_ranks')]
        table += f"|{a['id']}|{'|'.join(vals)}|{a['review_status']}|\n"
    explanation = '''# 重排名次下降核对（2026-11-16）

这是保存分数的确定性核对，没有重新检索、调用模型或调用 API；没有修改业务代码、Gold 或配置。纯模型/混合分列只是在同一批已评分候选中重新排序，不能直接解释为端到端采用效果。

## 结论

- 004：去重6→纯模型3→辅助混合9→排名保护9。模型本来把正确正文提前，辅助词规则把它压低。正确页描述“1#/2#/3#分别采集哪些传感器”，但没有“位置/安装/放置”等关系词，支持分仅0.3881。更换板卡的幻灯片写了“反应盘采集板位置”，支持分0.9060，从纯模型5升至最终1。规则把“板卡安装位置”和“板卡采集哪个位置的信号”混淆。
- 006：15→4→10→18。正确页写“低位触发启动、高位触发停止、空瓶自动切换”，支持分仅0.4725；装机检查页因含“操作/检查”等词获0.9197。之后排名保护为前融合名次候选让位：第4位强插融合1的CL-2000i缓冲容器空故障，第7位强插融合4的CL-2000i填充缓慢故障等，正确页融合15，到第18位才被选。保护继承上游排名，不判断对象是否适用。
- 007：3→2→7→7。参数表虽有型号、140±5及表头单位，却没有规则所列“参数/标准/范围”等词，支持分0.4550；“通过参数调整”的幻灯片获0.9025，题库仅要求口述标准也获0.9200。必须区分相关题干与实际参数答案。另有高速免疫第22张145±5 μL/s与表中140±5冲突，本题disputed，不认定第一名145±5已被证实错误。

当前分数=65%模型sigmoid分+5%检索名次倒数均值+30%词规则支持分，再执行排名保护。004模型正确页0.99345高于更换页0.98647，但辅助分差带来约0.15537的劣势，远大于模型加权优势0.00454。30%不是“小修正”；关系词单次奖励最高可影响混合分0.15。

排名保护参考的是原融合名次，不是去重后名次。006正确页本来混合第10，却被该保护进一步压至18。该保护并非保证每段都不会变差，而是保护先前靠前者，其它候选会被挤后。

## 边界

014还有模型自身后移（去重2→纯模型3）。023则辅助分改善7→6，但保护压到11。019场景页1→7→11→4；处理步骤页去重30，在Top20入口就没有获得模型分数，不能把这一页丢失归因于重排打分。排名下降也不等于整题失败：005等仍可保留充分证据；006装机指南49页至少包含部分同类有效事实，不能把所有超过Gold的候选都称作错误资料。

## 核对范围

下表覆盖主集合49题中第一条映射正向证据名次下降的13题；包含pending/disputed，仅作诊断；多正向片段逐条列排名，非完整覆盖率。完整候选分数、分项与保护决策在diagnosis.json。固定的91题全部评分和保护顺序逐一复现；本轮没有给出修复已有效的结论。

'''
    (OUT / 'conclusion.md').write_text(explanation + table, encoding='utf8')
    # Reuse the application's existing report viewer and verified frozen upstream matches.
    historical = (HISTORY / 'report.html').read_text(encoding='utf8')
    encoded = re.search(r"const ARMS=await unpack\('([^']+)'\)", historical)
    frozen = json.loads(gzip.decompress(base64.b64decode(encoded[1])))['A']
    upstream = {c['id']: c for c in frozen['cases']}
    original_match = service._match_stage_candidates
    def match(case, stage, candidates, config, formal_mappings=None):
        if stage not in ('context', 'evidence'):
            cached = upstream[case.case_id]['stage_data'][stage]['candidates']
            assert [x['chunk_id'] for x in cached] == [x['chunk_id'] for x in candidates[:config.topk]]
            return deepcopy(cached), {}
        return original_match(case, stage, candidates, config, formal_mappings)
    service._match_stage_candidates = match
    selected = {a['id'] for a in audits}
    dataset = load_gold_dataset(ROOT / 'tests/rag/fixtures/rag_gold_v5.jsonl', reviewed_only=False)
    dataset = GoldDataset(dataset.path, dataset.sha256, tuple(c for c in dataset.cases if c.case_id in selected))
    raw = []
    for r in rows:
        if r['id'] not in selected:
            continue
        a = production[r['id']]
        raw.append({**{k: r[k] for k in ('id', 'question', 'group_id', 'review_status', 'cohort')}, 'scores': a['scores'], 'stages': a['scores'], 'candidate_stages': {**r['candidate_stages'], 'evidence': a['evidence']}, 'gold_mappings': mappings[r['id']], 'events': [{'stage': 'query_interpreted', 'query_plan': a['query_plan']}, {'stage': 'context_built', 'context_entries': a['entries'], 'llm_context': a['context']}], 'answer': {}, 'gold_judgement': {}, 'actual_behavior': 'not_run'})
    print('Rendering 13-case native diagnostic report', flush=True)
    detail = service.build_report_detail(dataset, {'run_id': 'rerank_rank_diagnosis_20261116', 'ai_bundle_id': manifest['bundle'], 'mode': '保存分数核对；本轮未调用API'}, raw)
    sources = read(HISTORY / 'report_sources.json'); used = set(); unresolved = []
    def source_ids(record):
        path = str(record.get('file_path') or record.get('file') or '').replace('\\', '/')
        loc = str(record.get('location') or '')
        page = record.get('page_number') or record.get('page'); slide = record.get('slide_number')
        found = re.search(r'第\s*(\d+)\s*(页|张)', loc)
        if found and not page and not slide:
            if found[2] == '张': slide = int(found[1])
            else: page = int(found[1])
        if Path(path).suffix.lower() in ('.ppt', '.pptx') and not slide: slide, page = page, None
        if ' > ' in path:
            archive, member = path.split(' > ', 1)
            with ZipFile(archive) as z:
                names = []
                for info in z.infolist():
                    variants = [info.filename]
                    if not info.flag_bits & 0x800:
                        try: variants.append(info.filename.encode('cp437').decode('gbk'))
                        except (UnicodeError, LookupError): pass
                    if any(v.replace('\\', '/') == member for v in variants): names.append(info.filename)
                assert len(names) == 1, path
            location = {'location_text': loc, 'page_number': page, 'slide_number': slide}
            sid = hashlib.sha256((path + json.dumps(location, ensure_ascii=False)).encode()).hexdigest()[:24]
            sources[sid] = {'path': path, 'archive': archive, 'member': names[0], 'location': location}
            used.add(sid); return [sid]
        matches = [sid for sid, s in sources.items() if path and s['path'].replace('\\', '/').casefold().endswith(path.casefold()) and str(s['location'].get('page_number') or '') == str(page or '') and str(s['location'].get('slide_number') or '') == str(slide or '')]
        if len(matches) == 1:
            used.add(matches[0]); return matches
        unresolved.append({'path': path, 'location': loc}); return []
    for case in detail['cases']:
        for e in case.get('gold_evidence', []) + case.get('hard_negatives', []): e['source_ids'] = source_ids(e)
        for stage in case['stage_data'].values():
            for x in stage['candidates']: x['source_ids'] = source_ids(x)
    page = service.render_html(detail)
    table_html = '<table><tr><th>题号</th><th>去重</th><th>纯模型</th><th>加辅助分</th><th>排名保护后</th></tr>'
    for a in audits:
        table_html += '<tr><td>' + a['id'] + '</td>' + ''.join('<td>' + ','.join(map(str, a[k])) + '</td>' for k in ('dedup_positive_ranks', 'model_positive_ranks', 'combined_positive_ranks', 'final_positive_ranks')) + '</tr>'
    banner = '<section class="panel"><h2>重排名次下降：模型、辅助分与排名保护分开核对</h2><p>91题保存分数逐一复现。下方列主集合名次下降的13题，非新一轮问答，也非采用效果对照。Gold、代码、设置均未改。</p>' + table_html + '</table><details><summary>逐题原因与核对边界</summary><pre>' + html.escape(explanation) + '</pre></details><p id="sourceStatus">使用同目录open_report.cmd打开后可点击定位原资料页。</p><a href="conclusion.md">详细结论</a></section>'
    page = page.replace('<main>', '<main>' + banner, 1)
    js = '''
function sourceLinks(r){return (r.source_ids||[]).map(id=>`<button data-original-source="${esc(id)}">打开原资料对应页</button>`).join('');}
document.addEventListener('click',async e=>{const b=e.target.closest('[data-original-source]');if(!b)return;e.preventDefault();const s=document.getElementById('sourceStatus'),token=new URLSearchParams(location.search).get('token');if(!token){s.textContent='请使用open_report.cmd打开报告。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.originalSource})});const d=await r.json();s.textContent=d.message||d.error;}catch(err){s.textContent=String(err);}});
'''
    page = page.replace('<b>${esc(e.evidence_id)}</b>', '<b>${esc(e.evidence_id)}</b> ${sourceLinks(e)}')
    page = page.replace('<div class="candidate-body"><div class="inline-caption">状态：', '<div class="candidate-body">${sourceLinks(c)}<div class="inline-caption">状态：')
    assert '\ninit();\n</script>' in page
    page = page.replace('\ninit();\n</script>', '\n' + js + '\ninit();\n</script>')
    (OUT / 'report.html').write_text(page, encoding='utf8')
    write(OUT / 'report_sources.json', {key: sources[key] for key in sorted(used)})
    write(OUT / 'report_audit.json', {'cases': len(detail['cases']), 'source_locations': len(used), 'unresolved_locations': unresolved, 'candidate_formula_checks': verified, 'guard_replay_cases': len(rows), 'historical_answers_imported': False, 'api_calls': 0})
    cmd = f'@echo off\r\ncd /d "{ROOT}"\r\n"{ROOT / ".venv/Scripts/python.exe"}" -X utf8 -m tools.serve_rag_report --report "{OUT / "report.html"}"\r\n'
    (OUT / 'open_report.cmd').write_text(cmd, encoding='utf8')
    print(json.dumps({'cases': len(audits), 'formula_checks': verified, 'guard_checks': len(rows), 'source_locations': len(used), 'unresolved': len(unresolved)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
