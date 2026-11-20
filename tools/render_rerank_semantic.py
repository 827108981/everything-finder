"""Application-format rerank comparison with actual contexts and source-page links."""
from hashlib import sha256
import html
import json
import re
from types import SimpleNamespace
from pathlib import Path

from tools.validate_rerank_semantic import OUT,read_jsonl,ROOT
from tools.validate_selection_factorial import write
from tools.render_selection_factorial import main as render_native


LABELS={'baseline':'P · 当前混合分与排名保护','B':'B · 纯 BGE（诊断对照）',
    'R':'R · BGE + 结构证据规则','M':'M · BGE + 本地语义判别','RM':'RM · 规则 + 本地语义判别'}


def pending_html(sources):
    gold={r['id']:r for r in read_jsonl(OUT/'new_holdout_pending.jsonl')}
    arms={a:{r['id']:r for r in read_jsonl(OUT/('fresh_'+a+'.jsonl'))} for a in ('baseline','B','R')}
    cards=[]
    def source_button(raw,page=None,slide=None,location=''):
        if page is None and slide is None:
            match=re.search(r'第\s*(\d+)\s*(页|张)',location)
            if match:
                if match[2]=='页':page=int(match[1])
                else:slide=int(match[1])
        location={'location_text':location,'page_number':page,'slide_number':slide}
        path=str(Path(raw).resolve())
        assert Path(path).is_file(),path
        key=sha256((path+json.dumps(location,ensure_ascii=False,sort_keys=True)).encode()).hexdigest()[:24]
        sources[key]={'path':path,'location':location}
        return '<button type="button" data-original-source="'+key+'">打开原资料对应页</button>'
    for cid,case in gold.items():
        evidence=''.join('<p>'+source_button(e['file'],slide=int(e['page']),location=e['position'])+
            html.escape(e['file']+' · '+e['position'])+'</p><pre>'+html.escape(e['quote'])+'</pre>' for e in case['gold_evidence'])
        table='<table><thead><tr><th>方案</th><th>关键词</th><th>向量</th><th>融合</th><th>重排</th><th>条件</th><th>实际Context</th></tr></thead><tbody>'
        contexts=''
        for arm,rows in arms.items():
            r=rows[cid]
            def cell(stage):
                s=r['scores'][stage]
                return '<td>'+str(s['gold_evidence_found'])+'/'+str(s['gold_evidence_total_declared'])+'条；首条排名 '+str(s.get('first_gold_rank') or '未命中')+'</td>'
            table+='<tr><td>'+LABELS[arm]+'</td>'+''.join(cell(s) for s in ('lexical','vector','fusion','rerank','constraints','context'))+'</tr>'
            entries=''
            for e in r['entries']:
                buttons=''.join(source_button(s['file_path'],s.get('page_number'),s.get('slide_number'),s.get('location_text') or s.get('location','')) for s in e['sources'])
                entries+='<details><summary>'+html.escape(e['filename']+' · '+e['location'])+'</summary>'+buttons+'<pre>'+html.escape(e['included_text'])+'</pre></details>'
            contexts+='<div><h4>'+LABELS[arm]+'</h4>'+entries+'<details><summary>门禁与排除理由</summary><pre>'+html.escape(json.dumps(r['decision'],ensure_ascii=False,indent=2))+'</pre></details></div>'
        cards.append('<details><summary>'+cid+' · pending · '+html.escape(case['question'])+'</summary>'+table+'</tbody></table><h4>冻结引文</h4>'+evidence+
            '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px">'+contexts+'</div></details>')
    return '<section class="panel" id="freshPending"><h2>另列：12道新题（全部pending，不计正式分）</h2><p>26条原资料与索引引文文本已核对。三个方案都为10/12完整。08和11在条件门禁被清空；模型组件未过关，未进入独立采用验收。源文档与历史集可重叠，目标索引片段此前未用于91题候选。</p>'+''.join(cards)+'</section>'


def main():
    metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
    arms={a:SimpleNamespace(remove_document_cap=True,budget_pack=False) for a in LABELS if a in metrics}
    rows=''.join('<tr><td>'+LABELS[a]+'</td><td>'+str(m['formal']['complete'])+'/46</td><td>'+str(m['diagnostic']['complete'])+'/40</td><td>'+str(len(m['formal']['loss_ids']))+' / '+str(len(m['diagnostic']['loss_ids']))+'</td><td>'+str(m['formal']['new_negative_quote_count'])+' / '+str(m['diagnostic']['new_negative_quote_count'])+'</td></tr>' for a,m in metrics.items())
    banner='''<section class="panel"><h2>重排五组实验：只比较API前，不生成业务答案</h2>
    <p>冻结同一AI包、原问题和20个候选；仅修改重排顺序。条件检查、最终A选段（无单文档配额，Top8，12000字符）不变。API调用0；应用默认未切换。</p>
    <p>现有91题：50 reviewed，其中46题有正向引文；其余41题仅诊断，其中40题有正向引文。另有12道新题pending，单独展示、不混算。</p>
    <table><thead><tr><th>方案</th><th>正式完整证据</th><th>诊断完整证据</th><th>丢Gold题（正式/诊断）</th><th>新增已标负例引文（正式/诊断）</th></tr></thead><tbody>'''+rows+'''</tbody></table>
    <p><b>已确认：</b>B/R恢复004、006。R把023完整报警表提前，但相对B未新增完整证据题。本地模型连续两次把LVC处理资料判为可支持CSDB，组件安全门槛失败；M/RM只作诊断，不建议启用。</p>
    <p><b>组合对照结论：</b>M/RM相对B没有新增完整Gold证据题，却在002新增主控板D107/D108负例（用户问功率驱动板D63/D64）。该片段未送模型；其他片段被降级后，它从B的第9升至第8进入Context。这是部分候选判别造成的间接挤入，不能只检查被模型提权的片段。</p>
    <p>“Gold没有丢失”不等于没有错场景；部分跨型号干扰在基线已存在，新增与既有分别记录。019的解决页尚未进入这20个候选，不能记作重排修复。</p>
    <label>切换方案：<select id="armSelect">'''+''.join('<option value="'+a+'"'+(' selected' if a=='R' else '')+'>'+LABELS[a]+'</option>' for a in arms)+'''</select></label>
    <p id="sourceStatus">请通过同目录 open_report.cmd 打开，再点击按钮直接定位原资料页。直接打开HTML可查看离线对照。</p>
    <p><a href="#freshPending">查看12道新题</a> · <a href="conclusion.md">完整结论</a> · <a href="final_acceptance.json">验收记录</a></p></section>'''
    render_native(output=OUT,arms=arms,labels=LABELS,reader=lambda a:{r['id']:r for r in read_jsonl(OUT/(a+'.jsonl'))},
        mutable_stages=('rerank','constraints','evidence','context'),initial_arm='R',banner_override=banner)
    sources=json.loads((OUT/'report_sources.json').read_text(encoding='utf8'))
    pending=pending_html(sources)
    path=OUT/'report.html';page=path.read_text(encoding='utf8')
    page=page.replace('</main>',pending+'</main>',1)
    path.write_text(page,encoding='utf8');write(OUT/'report_sources.json',sources)
    audit=json.loads((OUT/'report_format_audit.json').read_text(encoding='utf8'))
    audit.update(source_locations=len(sources),pending_new_cases=12,pending_in_formal_metrics=False,
        html_bytes=len(page.encode()),mutable_stages=['rerank','constraints','evidence','context'])
    write(OUT/'report_format_audit.json',audit)


if __name__=='__main__':main()
