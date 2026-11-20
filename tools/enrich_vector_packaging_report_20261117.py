"""Presentation-only completion summary; leaves frozen experiment code untouched."""
import hashlib
import html
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rag_vector_packaging_20261116'

def main():
    acceptance=json.loads((OUT/'final_acceptance.json').read_text(encoding='utf8'))
    assert acceptance['completed'] and acceptance['api_calls']==0
    data=json.loads((OUT/'detailed_findings.json').read_text(encoding='utf8'))
    target=OUT/'report.html'
    page=target.read_text(encoding='utf8')
    original_hash=hashlib.sha256(page.encode()).hexdigest()
    # The native renderer lists diagnostic controls alphabetically. Start the
    # review on the completed baseline pipeline, not a vector-only control.
    page=re.sub(r'let activeArm="[^"]+";', 'let activeArm="W0_native";',page,count=1)
    page=page.replace('<option value="W0_native">','<option value="W0_native" selected>')
    old=(OUT/'pytest.log').read_text(encoding='utf8').splitlines()[-1]
    new=(OUT/'all_pytest.log').read_text(encoding='utf8').splitlines()[-1]
    page=page.replace(html.escape(old),html.escape(new))
    labels={'W0':'W0 原包装','W1':'W1 去文件名','W2':'W2 去位置','W3':'W3 只保留正文标签',
            'W4':'W4 纯正文','W5':'W5 可信精简背景','F10':'F10 原包装＋原生表格行',
            'F11':'F11 组合（本轮与F10相同）','F10_count_matched':'等数量替换敏感性'}
    rows=''
    for record in data['packaging']:
        m=record['exact'];a=record['ann']
        ann=f'{a["formal"]["complete"]}/46；{a["diagnostic"]["complete"]}/40' if a else '未满足进入条件'
        rows+=f'<tr><td>{labels[record["arm"]]}</td><td>{m["formal"]["complete"]}/46</td><td>{m["diagnostic"]["complete"]}/40</td><td>{ann}</td></tr>'
    context_rows=''
    for arm in ('W0','F10','F11'):
        m=data['native_stage_summary'][arm]['context']
        context_rows+=f'<tr><td>{labels[arm]}</td><td>{m["formal"]["complete"]}/46</td><td>{m["diagnostic"]["complete"]}/40</td><td>{m["source_holdout"]["complete"]}/20</td></tr>'
    remaining=''.join(f'<tr><td>{r["id"]}</td><td>{html.escape(r["question"])}</td><td>{r["context_found"]}/{r["total"]}</td><td>{html.escape(r["earliest_observed_loss"])}</td></tr>' for r in data['remaining'])
    panel=f'''<!-- detailed-comparison-start -->
    <section class="panel" id="completedComparison">
    <h2>最终结论：不更换默认包装，结构补充暂不进入生产索引</h2>
    <p>已完成全库精确对照、真实ANN、111题三组真实检索链路及69项单元/兼容测试。API调用0。保留原Gold和生产索引，新增20题仍为pending。</p>
    <p><b>表格补充确实改善向量召回，但没有增加本批题的实际Context证据。</b>“首样本启动时间”和“通信接口”两题从向量漏召回变为完整；ISE比较题只补齐2/3条引文。原关键词召回已覆盖这些新增内容，所以最终Context覆盖没有增加，也没有新增丢失。</p>
    <h3>全库表示对照：返回父块的完整Gold引文覆盖</h3>
    <table><tr><th>方案</th><th>正式完整@50（精确）</th><th>原诊断完整@50（精确）</th><th>真实ANN：正式；诊断</th></tr>{rows}</table>
    <p>W1—W5均有必要引文丢失，不采用。F10增加1279个原生PPT表格行；等数量替换下仍保留两道补齐收益，收益并非仅由向量数量增加解释。该替换是敏感性对照，不是无损发布方案。</p>
    <h3>实际Context：只核对真正保留的included_text</h3>
    <table><tr><th>方案</th><th>正式正向46题</th><th>原诊断40题</th><th>新来源20题</th></tr>{context_rows}</table>
    <p>新增独立来源题的向量覆盖均为20/20，但仅1题有本次结构机制触发机会，且基线已完整。没有独立收益，不能宣称泛化已经通过；也不能据多数未触发的问题断言结构机制无用。原问题不变，未通过修改Gold制造提升。</p>
    <h3>当前仍缺引文的问题及最早可观测环节</h3>
    <table><tr><th>题号</th><th>问题</th><th>实际Context覆盖</th><th>环节</th></tr>{remaining}</table>
    <p>019保留了第13张DEV场景，但第14张原因/处理页未进入重排输出前20。表格补充没有修复这个已有问题。此表是证据链诊断，不是答案正确率。</p>
    <p>新Top5的21题已作技术复核：部分新增行属于错误部件、其他型号或相邻参数，不能直接用作答案；“已标注负例没有增加”不等于所有候选适用。未调用LLM，因此不对答案幻觉或正确率作本轮结论。</p>
    <p><a href="detailed_conclusion.md">详细结论与逐题前后对照</a> · <a href="detailed_findings.json">完整对照数据</a> · <a href="new_top5_technical_applicability_review.json">新增Top5技术复核</a></p>
    <p>下方“选择方案”可在W0_native、F10_native、F11_native之间切换，逐题查看各阶段、原始片段、实际Context与原资料页按钮。请用open_report.cmd启动，才能定位原资料。</p>
    </section><!-- detailed-comparison-end -->'''
    page=re.sub(r'<!-- detailed-comparison-start -->.*?<!-- detailed-comparison-end -->','',page,flags=re.S)
    page=page.replace('<main>','<main>'+panel,1)
    target.write_text(page,encoding='utf8')
    audit=dict(experiment_changes=False,label_changes=False,original_report_sha256=original_hash,
               enriched_report_sha256=hashlib.sha256(page.encode()).hexdigest(),default_arm='W0_native',tests=69)
    (OUT/'report_presentation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(target)

if __name__=='__main__':main()
