"""Source-local scope observations, not domain facts or a semantic classifier."""
from __future__ import annotations
import re


def exclusive_scope(text, pattern):
    """Only an explicitly restricted source statement can authorize a conflict."""
    if not pattern: return []
    values=[]
    for line in text.splitlines():
        if re.search(r'仅适用|仅用于|只适用|^(?:适用型号|适用范围|型号|软件版本)\s*[:：]',line.strip()):
            values.extend(m.group() for m in re.finditer(pattern,line))
    return list(dict.fromkeys(values))


def observations(plan, text):
    if not plan.adapter_active:
        return []
    policy=plan.adapter_policy
    result=[]
    for kind in policy.get('scope_entity_types',[]):
        pattern=policy.get('patterns',{}).get(kind)
        matches=[(m, m.group()) for m in re.finditer(pattern,text)] if pattern else []
        for value,words in policy.get('aliases',{}).get(kind,{}).items():
            for word in words:
                matches.extend((m,value) for m in re.finditer(re.escape(word),text,re.I))
        for m,value in matches:
            row={'type':kind,'text':m.group(),'normalized_value':value,'start':m.start(),'end':m.end(),
                 'confidence':.85,'source':'indexed_text_span',
                 'reason':'Mention in this source only; not proof that every statement applies to it'}
            if row not in result: result.append(row)
    return result


def contract_summary(plan):
    if not plan.adapter_active: return ''
    text=('\n证据使用约束：逐条引用原文中的对象和适用范围。型号、产品线、试剂、校准品、软件版本和阶段'
          '只适用于其所述规则；用户没给型号也不能省略资料的型号限制。不要将局部规则说成所有型号通用。'
          '来源中的名词出现不代表该段所有事实都适用，必须结合相邻条件。'
          '子部件与整机/组件不是同一对象：说明原文维护或更换的具体对象；'
          '原文只有子部件周期时不能据此给整套组件更换周期，必要时说明缺口或澄清。'
          '缺少某条件不等于冲突。多种处理路径的问题，能依据原文解释条件分支就直接解释；'
          '只有必须替用户选定唯一方案而缺决定条件时才追问。')
    goals={s.name for s in plan.answer_requirements}
    if 'cause' in goals and not goals.intersection({'handling','procedure','comparison'}):
        text+=('\n本题只问可能原因：优先使用诊断标题和故障前提直接对应的原因段，逐项简要回答。'
               '未逐段写出型号不等于该一般原因不适用；按原资料范围说明，不推导专属参数。'
               '不要把其他故障现象的排查流程、更换件或阈值当成本题答案，也不要把可能原因说成已确认故障。')
    if any(s.name=='method' for s in plan.strategies) and not ('cause' in goals and not goals.intersection({'handling','procedure','comparison'})):
        text+=('\n这是方法问题：保留原文中与用户所问有关的判断条件、处理分支和操作顺序，不增加未问到或无依据的阈值。'
               '术语解释不能替代处理流程；没有原文依据不要推断某次操作或某个部件一定有问题。'
               '解释方法不要求实际计算，不要因此强制追问。只有用户要求计算且缺原始数据时，才说明具体缺什么。')
    return text


def context_audit(plan, entries):
    """Deterministic presence checks; explicitly not entailment or answer grading."""
    actual='\n'.join(e.get('included_text','') for e in entries)
    scopes=[]
    for e in entries:
        for item in e.get('query_plan_diagnostics',{}).get('scope_observations',[]):
            scopes.append({**item,'source_id':e['source_id'],
                           'retained':item['text'] in e.get('included_text','')})
    return {'active':plan.adapter_active,'scope_observations':scopes,
            'condition_markers_present':bool(re.search(r'如果|若|允许|超过|上次|本次|前提',actual)),
            'step_markers_present':bool(re.search(r'步骤|重新|先|然后|之后|解决思路',actual)),
            'ambiguities':[{'name':s.name,'reason':s.reason} for s in plan.ambiguities],
            'semantic_completeness':'unverified',
            'reason':'Text-presence diagnostics only; source binding and final claims require separate verification'}
