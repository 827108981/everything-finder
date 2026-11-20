"""Produce auditable validation report without revising historical scores."""
import json,html
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/rag_scene_entry_20261105'

def main():
 rows=[json.loads(l) for l in (OUT/'cases.jsonl').read_text(encoding='utf8').splitlines()]
 probes=[json.loads(l) for l in (OUT/'applicability_outputs.jsonl').read_text(encoding='utf8').splitlines()]
 assert len(rows)==78 and len(probes)==12
 summary=json.loads((OUT/'summary.json').read_text(encoding='utf8'))
 groups=json.loads((OUT/'050_scoring_audit.json').read_text(encoding='utf8'))
 cohorts=[('原18固定集合（有效17）',[r for r in rows if r['cohort']=='formal48' and int(r['id'].split('-')[-1])<=18]),('新增集合（有效31）',[r for r in rows if r['cohort']=='formal48' and int(r['id'].split('-')[-1])>18])]+[(c,[r for r in rows if r['cohort']==c]) for c in ('regression10_pending','used_holdout10_pending','new_holdout10_pending')]
 lines=['# 场景关联与等价证据验证报告','',
 '**结论：050存在指定来源过窄导致的漏计；019场景正文已独立召回并排到第一，但仍被单文档3段配额挡住，最终无改善。通用适用性提示的回答实验结果见下，不以局部变化宣称整体质量提升。**','',
 '## 固定条件和实验区别','',
 '索引g000033-71ec3c52f3；模型gpt-5.6-sol；原有设置、Gold、H8及发布索引不变。本轮新增可选评测适配器和独立实验脚本。没有切换应用默认评分或部署检索/生成改动。',
 '旧A依赖原候选中的标题页；本轮独立检索现有5组真实连续页。旧C替换/合并候选曾丢021分支；本轮只在空位追加原物理页，不替换任何原上下文，完整计入8段、12000字和每文档3段上限。旧文件名标题前置实验改变CE输入；本轮关联的是真实相邻页场景和处理正文，保留原页来源。',
 '78题复放上轮保存的原上下文，单独执行新增检索、CE和条件检查；不是78题新跑完整回答。两个旧留出集合已经参与分析，本轮不能称为独立盲测。','',
 '## 050判分修正','',
 '批准的来源按OR计分，任一完整支持来源即可；同一替代方案的多部分仍按AND。只核对实际included_text、来源ID和完整引文，不接受同页其他行、父片段全文回退、错误来源或仅相似文字。业务批准范围仅为四类通用检查方向，不批准迁移机型专属参数。4份来源的引文均在当前索引及原PPT（3份）或已核实且哈希未变的原PDF（1份）中复核。',
 '| 保存上下文 | 原指定PDF命中 | 已批准四类检查证据组 |','|---|---:|---:|']
 for g in groups:lines.append(f"| {g['arm']} | {g['old']['gold_evidence_found']}/1 | {g['approved_group']['found']}/1 |")
 lines+=['','这是证据覆盖的漏计修正，不代表回答整体合格，也不是检索算法提升。历史Gold和分数未重写；评测适配器为显式可选，未自动推广其他语义等价关系。','',
 '## 019及回归证据检查','',
 '| 固定集合 | 题数 | 上下文改变 | 原证据丢失 | 新增已标注负例 |','|---|---:|---:|---:|---:|']
 for name,rs in cohorts:lines.append(f"| {name} | {len(rs)} | {sum(r['changed'] for r in rs)} | {sum(bool(r['lost']) for r in rs)} | {sum(bool(r['new_negatives']) for r in rs)} |")
 lines+=['',
 '019的真实13—14张连续页：影子BM25第2位；CE重排第1位，原始分0.997642，混合分0.925286。其原因和“建议灌注次数：40次以上”正文完整；原上下文已占满该文档3段，因此以document_cap3拒绝补入。没有临时放宽配额。019完整Gold引文命中仍为0；021仍保留原来2条完整引文，必要分支未再被挤掉。',
 '检索路线证明正确内容可找到，但最终选择未变；原上下文保留通过不等于方案有效。本轮019回答生成调用为0，因为输入逐字相同，按预设停止条件不重复生成来制造得分变化。','',
 '## 通用适用性提示的固定上下文回答验证','',
 '050及旧留出BS-800参数题各比较原提示与同一通用适用性规则，每条件3次，共12次。规则不含问题编号、机型名称、答案数值或四类原因。参数题用来检查是否把其他机型数字移用于提问对象；该题自身适用范围仍有待业务核实，不报告其正式正确率。无质量重试，原始输出和校验失败均保留。',
 f"12份输出中，引用校验失败{sum(bool(p.get('error')) for p in probes)}份；详见applicability_outputs.jsonl与manual_audit.json。引用ID有效不等于事实已获支持。",'',
 '| 观察维度（非整题正确率） | 原提示3次 | 通用适用性规则3次 |','|---|---:|---:|',
 '| 050明确覆盖四类通用检查方向 | 2/3 | 1/3 |',
 '| 050拒绝仅据暗计数失败直接确定更换件 | 3/3 | 3/3 |',
 '| 参数对照题以其他机型约1.14kΩ作为目标答案 | 2/3 | 1/3 |',
 '| 参数对照题引用校验失败 | 0/3 | 1/3 |','',
 '人工按四方向分别检查UPS/接地、漏光、测光位发光异物、光学模块异常。单写污染不替代明确的发光异物，背景中的模块性能确认不自动算作异常原因。050原提示第1份缺明确光学异常检查；规则第1份同样缺该项，规则第2份转向大量背景和条件操作，未覆盖供电/接地、发光异物等。参数题已有适用性争议，只记录可观察的跨机型数值引用与协议失败，不将其他输出自动记正确。',
 '结论：通用规则未获得稳定收益，050完整检查方向覆盖反而减少；参数题仍有不当数值采用和1次引用失败。因此本轮不建议启用该生成提示。3次重复只用于暴露波动，不作统计显著性或全量质量估计。','',
 '## 后续决定','',
 '保留050经业务批准的等价证据组作为本次修正依据。019本轮原型不启用；若继续，下一项应独立验证按场景分配同文档证据名额，而不是继续增加召回或更换重排模型。必须保留原始证据、满足同样总预算，并检查各条件分支；现有结果不能证明调整配额对全体问题安全。',
 '通用适用性提示是否采用，以逐份输出审核结果为准；不以无上下文变化的78题给出新的全量回答通过率。H8继续冻结。','',
 '## 验证与附件','',
 'RAG回归：302 passed、1 skipped；git diff --check通过。源文件、设置和冻结输入哈希前后相同。新OR判分测试覆盖正确来源、错误来源、同页错误正文、多片段缺失和未批准组。',
 '本目录保存protocol.json、scene_index.json、cases.jsonl、summary.json、050_approved_alternatives.json、original_quote_verification.json、050_scoring_audit.json，以及适用性实验全部输入输出。']
 text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text,encoding='utf8');(ROOT/'docs/rag_scene_entry_validation_20261105.md').write_text(text,encoding='utf8')
 (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>场景关联与等价证据验证</title><style>body{max-width:1100px;margin:32px auto;font:16px/1.65 system-ui}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>',encoding='utf8')
 (OUT/'verification.json').write_text(json.dumps({'context_replays':len(rows),'fresh_generations':len(probes),'citation_failures':sum(bool(p.get('error')) for p in probes),'summary':summary,'tests':'302 passed,1 skipped','diff_check':'passed'},ensure_ascii=False,indent=2),encoding='utf8')
 print('report generated')
if __name__=='__main__':main()
