"""Bounded exact-identifier preference and conservative, source-offset row evidence.

Only regular, complete labelled tables are projected. Ambiguous layout stays intact.
Coverage detects explicit structural support, not semantic entailment.
"""
from __future__ import annotations

import re
from local_full_text_search.rag.models import QueryPlan

HEADERS = ({'标识','报警代码','代码','identifier','code'},
           {'描述','含义','description','meaning'}, {'原因','cause'},
           {'解决方法','处理方法','解决措施','solution','handling'})
CELL_ID = re.compile(r'[A-Z][A-Z0-9_-]{1,20}')

def exact(text: str, identifier: str) -> bool:
    return bool(re.search(r'(?<![A-Za-z0-9_])'+re.escape(identifier)+r'(?![A-Za-z0-9_])',text,re.I))

def identifiers(plan):
    if not plan.adapter_active: return ()
    return tuple(dict.fromkeys(e.text for e in plan.entity_details
                 if e.type in ('alarm_code','instrument_model','material_code') and e.confidence>=.9
                 and e.strength not in ('soft','ambiguous')))

def structured_rows(text):
    lines=[(m.group().strip(),m.start()+len(m.group())-len(m.group().lstrip()),
            m.end()-len(m.group())+len(m.group().rstrip()))
           for m in re.finditer(r'[^\n\r]+',text) if m.group().strip()]
    for i in range(len(lines)-3):
        if not all(lines[i+j][0].casefold() in h for j,h in enumerate(HEADERS)): continue
        rest=lines[i+4:]
        # Page numbers are outside the table, never cells.
        if rest and rest[-1][0].isdigit(): rest=rest[:-1]
        if not rest or len(rest)%4: continue
        rows=[rest[j:j+4] for j in range(0,len(rest),4)]
        if not all(CELL_ID.fullmatch(r[0][0]) and all(c[0] for c in r[1:]) for r in rows): continue
        return [{'identifier':r[0][0],'start':r[0][1],'end':r[-1][2],
                 'prefix_start':0,'prefix_end':lines[i+3][2],
                 'quote':text[r[0][1]:r[-1][2]],
                 'slots':{name:{'quote':c[0],'start':c[1],'end':c[2]}
                          for name,c in zip(('definition','cause','handling'),r[1:])},
                 'confidence':.95,'source':'indexed_regular_table','reason':'Complete labelled four-column table; every row has identifier plus three nonempty cells'} for r in rows]
    return []

def annotate(plan, item):
    if not plan.entity_details: return {}
    if not plan.adapter_active:
        item.query_plan_diagnostics={'adapter_active':False,'identifier_conflict':False}
        return item.query_plan_diagnostics
    from local_full_text_search.rag.evidence_contract import observations,exclusive_scope
    text=item.chunk.text
    ids=identifiers(plan)
    matches=[value for value in ids if exact(text,value)]
    rows=[r for r in structured_rows(text) if any(exact(r['identifier'],v) for v in matches)]
    alarms=[e.text for e in plan.entity_details if e.type=='alarm_code']
    declared=re.findall(r'(?:报警|报|错误码|故障码)\s*[:：]?\s*([A-Z][A-Z0-9_-]{1,15})(?![A-Za-z0-9_])',text)
    # A heading such as "XYZ排查思路" explicitly names the alarm too;
    # absence of the word 报警 must not make another alarm's method generic.
    declared += re.findall(r'(?m)^\s*([A-Z][A-Z0-9_-]{1,15})\s*(?:排查思路|故障排查|处理方法|原因分析)',text)
    # Unknown identifiers are never hard exclusions. Only explicit target alarms
    # versus a different explicitly named alarm can be rejected.
    mismatch=bool(alarms and declared and not any(exact(text,a) for a in alarms))
    scope_conflicts=[]
    for kind in ('instrument_model','software_version'):
        pattern=plan.adapter_policy.get('patterns',{}).get(kind,'')
        declared_values=exclusive_scope(text,pattern)
        requested=[e for e in plan.scope_constraints if e.type==kind and e.strength not in ('soft','ambiguous')]
        if len(requested)==1 and len(declared_values)==1:
            normalize=lambda value: re.sub(r'\s|软件版本|版本号|软件|[:：]','',value).casefold()
            if normalize(requested[0].text)!=normalize(declared_values[0]):
                scope_conflicts.append({'type':kind,'requested':requested[0].text,'source_value':declared_values[0],
                    'reason':'Source explicitly restricts applicability; not inferred from a mere mention'})
    covered={key for r in rows for key in r['slots']}
    retained={k:v for k,v in item.query_plan_diagnostics.items() if k in ('matched_list_fields','source_consensus','source_representative','reservation_reason')}
    item.query_plan_diagnostics={
        **retained,
        'adapter_active':True,
        'scope_observations':observations(plan,text),
        'explicit_scope_conflicts':scope_conflicts,
        'exact_matches':matches,'structured_rows':rows,
        'identifier_conflict':mismatch, 'other_explicit_alarms':declared if mismatch else [],
        'protected_from_document_quota':False,
        'slot_coverage':[{ 'slot':s.name,'status':'covered' if s.name in covered else 'unknown',
            'confidence':.95 if s.name in covered else 0.,'source':'indexed_regular_table' if s.name in covered else 'none',
            'reason':'Nonempty labelled cell bound to the exact identifier' if s.name in covered else 'No structurally verified support; may require semantic review'} for s in plan.answer_requirements],
    }
    return item.query_plan_diagnostics

def prioritize(plan, items):
    if not plan or not plan.adapter_active: return list(items)
    for item in items: annotate(plan,item)
    # One representative per distinct identifier, not an unlimited boost for all
    # pages that mention a popular code/model. Other candidates keep original order.
    representatives=[]
    method=any(s.name=='method' for s in plan.strategies)
    cap=plan.adapter_policy.get('max_exact_representatives',2)
    for value in identifiers(plan):
        if len(representatives)>=cap: break
        hits=[i for i in items if value in i.query_plan_diagnostics.get('exact_matches',[])]
        if not hits: continue
        row=next((i for i in hits if i.query_plan_diagnostics.get('structured_rows')),hits[0])
        chosen=(next((i for i in hits if not i.query_plan_diagnostics.get('structured_rows')
            and re.search(r'如果|若|步骤|流程|排查思路|上次|本次|先.{1,40}再',i.chunk.text)),row)
                if method else row)
        if chosen not in representatives: representatives.append(chosen)
    for i in items:
        i.query_plan_diagnostics['exact_representative']=i in representatives
        i.query_plan_diagnostics['preference_reason']=('Method question: original relative order retained, no row promotion' if method else
            'Bounded exact representative; regular labelled row first' if i in representatives else 'Original relative order retained')
        i.query_plan_diagnostics['representative_cap']=cap
    return list(items) if method else representatives+[i for i in items if i not in representatives]

def coverage(plan, entries):
    result=[]
    for requirement in plan.answer_requirements:
        supports=[]
        for e in entries:
            for r in e.get('query_plan_diagnostics',{}).get('structured_rows',[]):
                cell=r['slots'].get(requirement.name)
                if cell and cell['quote'] in e.get('included_text','') and exact(e.get('included_text',''),r['identifier']):
                    supports.append({'source_id':e['source_id'],'identifier':r['identifier'],**cell})
        result.append({'requirement':requirement.name,'status':'covered' if supports else 'unknown',
                       'supports':supports,'confidence':.95 if supports else 0.,'source':'actual_context',
                       'reason':'Exact identifier and labelled cell survive context budget' if supports else 'No verified structural coverage; do not infer missing facts'})
    return result
