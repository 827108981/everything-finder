"""Compare pinned reranker outputs without modifying Gold."""
from __future__ import annotations
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT=Path('reports/rerank_models_20261102')


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    baseline=[json.loads(s) for s in Path('reports/rerank_title_input_20261102_threads4/cases.jsonl').open(encoding='utf-8')]
    original=[json.loads(s) for s in Path('reports/rerank_dev_holdout_20261101/cases.jsonl').open(encoding='utf-8')]
    original={r['id']:r for r in original if r['arm']=='pure'}
    rows={'base':{r['id']:{'scores':r['arms']['body']['scores'],
        'seconds':r['arms']['body']['inference_seconds'], 'order':r['arms']['body']['order'],
        'candidates':{cid:{'pair_tokens':tokens,'visible_body_end':r['arms']['body']['body_visible_end'][cid]}
            for cid,tokens in r['arms']['body']['untruncated_pair_tokens'].items()}} for r in baseline}}
    for model in protocol['models']:
        done=json.loads((ROOT/(model+'_completion.json')).read_text(encoding='utf-8'));assert done['n']==60
        rows[model]={r['id']:r for r in [json.loads(s) for s in (ROOT/(model+'_cases.jsonl')).open(encoding='utf-8')]}
    assert all(set(x)==set(original) for x in rows.values())
    for model,items in rows.items():
        for cid,row in items.items():
            assert len(row['order'])==len(set(row['order']))==20,(model,cid)
            if model!='base':
                assert all(math.isfinite(c['score']) for c in row['candidates'].values()),(model,cid)
    for d in protocol['datasets'].values():assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
    cohorts={'development':lambda r:r['cohort']=='development',
        'original18':lambda r:r['cohort']=='development' and int(r['id'][-3:])<=18,
        'added32':lambda r:r['cohort']=='development' and int(r['id'][-3:])>18,
        'holdout_pending':lambda r:r['cohort']=='holdout_pending'}
    summary={};changes={}
    for cohort,accept in cohorts.items():
        ids=[cid for cid,r in original.items() if accept(r) and r['expected_behavior']!='refuse']
        summary[cohort]={}
        for model,items in rows.items():
            summary[cohort][model]={}
            for k in ('1','3','8'):
                scores=[items[cid]['scores'][k] for cid in ids]
                groups=defaultdict(list)
                for cid in ids:groups[original[cid]['group_id']].append(items[cid]['scores'][k]['gold_evidence_recall'])
                summary[cohort][model][k]={'n':len(ids),'hit':sum(s['gold_hit'] for s in scores),
                    'coverage':sum(s['gold_evidence_recall'] for s in scores)/len(ids),
                    'mrr':sum(s['reciprocal_rank'] for s in scores)/len(ids),
                    'negatives':sum(s['hard_negative_found'] for s in scores),
                    'group_coverage':sum(sum(g)/len(g) for g in groups.values())/len(groups)}
    for model in protocol['models']:
        changes[model]=[]
        for cid,r in original.items():
            a=rows['base'][cid]['scores'];b=rows[model][cid]['scores']
            if a['20']['first_gold_rank']!=b['20']['first_gold_rank'] or a['8']!=b['8']:
                changes[model].append({'id':cid,'cohort':r['cohort'],'question':r['question'],
                    'base_rank':a['20']['first_gold_rank'],'model_rank':b['20']['first_gold_rank'],
                    'base_coverage':a['8']['gold_evidence_recall'],'model_coverage':b['8']['gold_evidence_recall'],
                    'new_negatives':sorted(set(b['8']['hard_negative_ids'])-set(a['8']['hard_negative_ids'])),
                    'removed_negatives':sorted(set(a['8']['hard_negative_ids'])-set(b['8']['hard_negative_ids']))})
    costs={};visibility={}
    for model,items in rows.items():
        costs[model]={'scoring_seconds':sum(r['seconds'] for r in items.values()),
            'median_query_seconds':median(r['seconds'] for r in items.values()),
            'truncated_pairs':sum(c['pair_tokens']>512 for r in items.values() for c in r['candidates'].values()),
            'body_characters_visible':sum(c['visible_body_end'] for r in items.values() for c in r['candidates'].values())}
        if model=='base' and (ROOT/'base_memory_probe.json').exists():
            costs[model]['separate_memory_probe_bytes']=json.loads((ROOT/'base_memory_probe.json').read_text(encoding='utf-8'))['rss_bytes']
        if model!='base':costs[model]['peak_sampled_rss_bytes']=max(r['peak_rss_bytes'] for r in items.values())
        if model!='base':
            losses=[];gains=[]
            for source in baseline:
                for quote in source['single_chunk_quote_audit']:
                    cid=quote['chunk_id'];new_visible=quote['char_end_needed']<=items[source['id']]['candidates'][cid]['visible_body_end']
                    entry={'id':source['id'],'chunk_id':cid,'file':quote['file'],'quote':quote['quote']}
                    if quote['body_visible'] and not new_visible:losses.append(entry)
                    elif not quote['body_visible'] and new_visible:gains.append(entry)
            visibility[model]={'newly_occluded':losses,'newly_visible':gains}
    refusal={model:{cid:row['scores']['8']['hard_negative_ids'] for cid,row in items.items()
                    if original[cid]['expected_behavior']=='refuse'} for model,items in rows.items()}
    result={'protocol':protocol,'summary':summary,'changes':changes,'costs':costs,'gold_input_visibility':visibility,
            'refusal_negative_diagnostics':refusal}
    (ROOT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    names={'base':'BGE base','bge_v2_m3':'BGE v2-m3','qwen3_06b':'Qwen3 0.6B'}
    labels={'development':'开发集46题','original18':'原18题中的16题','added32':'新增32题中的30题','holdout_pending':'候选10题pending'}
    lines=['# 三种本地重排模型的固定候选对照','', '## 结论','',
        '在本机CPU、FP32、512 token和固定候选条件下，当前结果不支持把BGE base型号替换为BGE v2-m3或Qwen3-Reranker-0.6B。BGE v2-m3前3条命中多1题，但首位命中少4题、指定反例次数增加，计算耗时约3.1倍。Qwen开发集首位、前3、前8均退步，计算耗时约5.6倍。此结论是当前部署预算与本地数据上的取舍，不代表模型在原生长上下文、其他领域或优化引擎下的上限。',
        '', '10道pending候选题中，新模型的首位指定文件命中较高，但三者前8条均为10/10；等价来源尚未标全，不能据此宣称新模型改善了业务回答。所有题均已用于此前检查，本轮也不是独立泛化验收。',
        '', '建议继续保留BGE base作为模型型号基线。现有应用的辅助分和名次保护另有已确认问题，本轮纯模型对照不能替它们作有效性背书。',
        '', '## 实验范围','',
        '同一AI包、同一60题、每题20条正文候选；CPU、float32、4线程、batch=4、总输入上限512 token。使用官方模型评分方式，只按模型分排序；不叠加规则分或名次保护。基线复用上一轮已实际计算、原排序复现的1200对；另两个模型各实际计算1200对。',
        '', 'Qwen使用官方通用检索指令及yes/no评分模板；模板计入512预算。同一token上限并非同一正文字符数；各模型分词不同，以下是固定部署预算的效果对照，不能完全分离模型能力与输入可见范围。不同模型分数不作数值高低比较，不共用相关性阈值。',
        '', 'Gold按已有完整引文计分，仍可能漏标等价来源。10道候选仍为pending；全部题目均已参与此前检查，不能称独立验收。未生成答案，也未复跑条件检查、文档限额或最终上下文。',
        '', '## 证据排序','', '| 集合 | 模型 | 首位命中 | 前3命中 | 前8命中 | 前8平均覆盖 | MRR@8 | 前8指定反例次数 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort in labels:
        for model in rows:
            d=summary[cohort][model];s=d['8']
            lines.append(f"| {labels[cohort]} | {names[model]} | {d['1']['hit']}/{s['n']} | {d['3']['hit']}/{s['n']} | {s['hit']}/{s['n']} | {s['coverage']:.1%} | {s['mrr']:.4f} | {s['negatives']} |")
    lines+=['','上表开发集为46道有正向证据的题；另4道refuse题不计入正向命中。其指定反例进入前8条的次数（风险诊断，非错误回答数）：']
    for model,cases in refusal.items():lines.append(f"- {names[model]}：{sum(len(v) for v in cases.values())}次。各题详情见完整统计。")
    lines+=['','## 本机成本和截断','', '| 模型 | 1200对分词＋推理总秒数 | 每题中位秒数 | 超512 token的输入对 | 工作集GiB |', '|---|---:|---:|---:|---:|']
    for model,c in costs.items():
        peak=f"{c['peak_sampled_rss_bytes']/1024**3:.2f}" if 'peak_sampled_rss_bytes' in c else '未采集'
        if 'separate_memory_probe_bytes' in c:peak=f"{c['separate_memory_probe_bytes']/1024**3:.2f}（单批探测）"
        lines.append(f"| {names[model]} | {c['scoring_seconds']:.1f} | {c['median_query_seconds']:.2f} | {c['truncated_pairs']}/1200 | {peak} |")
    lines+=['','耗时不含下载、加载和诊断审计；不同模型顺序运行，系统负载可能变化，不能直接当作生产延迟承诺。此次是Transformers CPU FP32参考实现，未做量化或专门推理引擎优化。新模型工作集为全程批次结束后采样最大值；base为相同数据加载环境下、4条实际最长候选截到512 token的单批补测，未测全程峰值，两者不可作严格峰值对比。均含运行时开销。',
        '', '仅对能在单个候选中逐字定位的完整Gold引文检查模型实际看到的正文字符边界；跨块引文未按此方式判断：']
    for model,d in visibility.items():
        lines.append(f"- {names[model]}：相对基线新增不可完整看到的引文{len(d['newly_occluded'])}处；新增可完整看到的引文{len(d['newly_visible'])}处。详情见完整统计。")
    lines+=['', '## 关键逐题复核','',
        '- Qwen在028题把唯一已命中的临床结果系统偏低正文由第2降到第10，前8条出现定标异常、通用板卡排查等不同场景。直接证据所需103个原文字符在base与Qwen输入中均完整可见，因此这次退步不能归因于截断。',
        '- BGE v2-m3新增002题的主控板D107资料（问题问功率驱动板D63/D64），属于不同板卡；031题新增的±2SD背景片段只相关，不能单独替代警告/失控规则。反例次数不是错误回答数。',
        '- BGE v2-m3与base看到的正文范围完全一致，338对发生截断；Qwen因分词及官方模板不同，有408对截断，额外遮住024及035题中的3处完整引文。Qwen对照存在可见范围差异，不能把所有变化都归因于模型能力。',
        '- 001、010候选题的其他教材包含相同关键事实，002另有手册解释核心功能；其指定文件名次变化不自动等于事实证据优劣。']
    lines += [
        '', '## 逐题变化','']
    for model,items in changes.items():
        lines += [f'### {names[model]} 相对 BGE base','', '| 题号 | 首条完整标注证据排名：基线/新模型 | 前8覆盖：基线/新模型 | 新增指定反例 |', '|---|---:|---:|---|']
        for r in items:
            f=lambda x:'—' if x is None else f'{x:.1%}'
            lines.append(f"| {r['id']} | {r['base_rank'] or '—'}/{r['model_rank'] or '—'} | {f(r['base_coverage'])}/{f(r['model_coverage'])} | {','.join(r['new_negatives']) or '—'} |")
    lines+=['','## 版本与资料','']
    for model,(repo,sha) in protocol['models'].items():lines.append(f'- `{repo}@{sha}`')
    lines += ['- [BGE官方用法](https://huggingface.co/BAAI/bge-reranker-v2-m3)',
        '- [Qwen官方用法](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)',
        f"- [冻结协议]({(ROOT/'protocol.json').resolve().as_posix()})",f"- [完整统计]({(ROOT/'comparison.json').resolve().as_posix()})"]
    Path('docs/rag_rerank_models_comparison_20261102.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'costs':costs},ensure_ascii=False))


if __name__=='__main__':main()
