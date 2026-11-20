"""Native four-arm applicability report, with current adoption decision."""
import html
import json
from tools.validate_condition_binding import OUT, ARMS
from tools.selection_factorial import ARMS as SELECTION
from tools.render_selection_factorial import main as render


def main():
    labels={'baseline':'基线 · 已采用A','extraction':'仅修改条件提取','judgement':'仅修改局部判断','combined':'提取与判断组合'}
    metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
    rows=''.join(f'<tr><td>{labels[a]}</td><td>{m["formal"]["complete"]}/46</td><td>{m["diagnostic"]["complete"]}/40</td><td>{html.escape("、".join(m["formal"]["loss_ids"]) or "无")}</td><td>{m["formal"]["existing_negative_quote_count"]}</td></tr>' for a,m in metrics.items())
    banner=f'''<section class="panel"><h2>条件检查四组对照：实验完成，组合方案暂不采用；正式仍为A</h2>
<p>91题 × 4组冻结候选重放；基线复现91/91；API调用0。50题已审核，其中46题有正向证据；另外40题有正向证据，仅作诊断。</p>
<p>四组均保留A选段、TopK8、12000字符。关键词/向量/融合/重排不变。组合方案只改变019和022的Context；不是答案正确率实验。</p>
<table><thead><tr><th>方案</th><th>正式题全部Gold引文覆盖</th><th>诊断题全部Gold引文覆盖</th><th>Gold丢失题</th><th>正式Context负例引文数</th></tr></thead><tbody>{rows}</tbody></table>
<p><b>采用阻断：019第49张是包含多项目与单项目的完整方法说明，被误判为单项目场景而整段删除。该页并非本题计分Gold，自动覆盖分数没有体现误删。</b>022明确的个别项目DEV/LVC片段得到排除；应保留该局部收益记录，但不能掩盖019误删。</p>
<p>真实AnswerService.retrieve入口4题×2组，无API；8道新增资料检查均未误删，但主要证明保留能力，尚未证明跨业务冲突识别。新开关仅构造器注入，默认关闭，持久化设置及Gold不变。</p>
<label>选择对照方案：<select id="armSelect">{''.join(f'<option value="{a}"'+(' selected' if a=='combined' else '')+f'>{labels[a]}</option>' for a in ARMS)}</select></label>
<p id="sourceStatus">双击同目录open_report.cmd，再点击按钮可定位原资料页。直接打开HTML可离线查阅。</p>
<a href="conclusion.md">完整结论与限制</a> · <a href="source_audit.json">原文与索引核验</a> · <a href="independent_results.json">新增资料检查</a> · <a href="native_audit.json">真实入口验证</a>
</section>'''
    def reader(a):return {r['id']:r for r in map(json.loads,(OUT/(a+'.jsonl')).read_text(encoding='utf8').splitlines())}
    # All selection factors equal the adopted A. Only applicability varies.
    render(output=OUT,arms={a:SELECTION['A'] for a in ARMS},labels=labels,reader=reader,
           mutable_stages=('constraints','evidence','context'),initial_arm='combined',banner_override=banner)


if __name__=='__main__':main()
