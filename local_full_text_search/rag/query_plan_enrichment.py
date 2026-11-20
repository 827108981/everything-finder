"""Generic, source-grounded question parsing. No corpus/Gold access."""
from __future__ import annotations

from dataclasses import replace
import re

from local_full_text_search.rag.domain_adapter import DomainAdapter
from local_full_text_search.rag.models import (
    QueryEntity, PlanSignal, SubQuery, QueryConstraint, ConstraintKind,
    ConstraintStrength, ConstraintPolarity, QueryPlan,
)

IDENTIFIER = r'(?<![A-Za-z0-9_])[A-Z][A-Z0-9]{1,15}(?![A-Za-z0-9_])'
MIXED_IDENTIFIER = r'(?<![A-Za-z0-9_])[A-Za-z]+[-_]?[0-9]+[A-Za-z0-9-]*(?![A-Za-z0-9_])'
REQUIREMENTS = {
    'definition': r'什么意思|是什么|含义|表示什么|代表什么',
    'cause': r'为什么|原因|怎么回事|什么导致',
    'handling': r'怎么处理|如何处理|怎么办|怎么解决|如何解决|怎么排查|如何排查|检查什么|检查哪些|优先检查|如何判断|怎么判断|什么时候|什么情况下',
    'procedure': r'步骤|先后|顺序|如何操作|怎么操作|流程',
    'comparison': r'比较|对比|区别|差异|分别|相比',
    'evidence_requirement': r'依据|证据|出处|引用|来源',
}

def signal(name, reason, confidence=1.0, source='question_rule'):
    return PlanSignal(name, confidence, source, reason)

def surface_status(text, start, end):
    if any(m.start() <= start and end <= m.end() for m in re.finditer(r'“[^”]*”|"[^"]*"|「[^」]*」',text)):
        return ConstraintPolarity.UNKNOWN, ConstraintStrength.AMBIGUOUS
    prefix=re.split(r'[，,。；;?？]',text[max(0,start-12):start])[-1]
    if re.search(r'(?:不是|并非|不属于)\s*$',prefix):
        return ConstraintPolarity.NEGATED,ConstraintStrength.AMBIGUOUS
    if re.search(r'可能|疑似|好像|还是|相比',prefix):
        return ConstraintPolarity.AFFIRMED,ConstraintStrength.SOFT
    return ConstraintPolarity.AFFIRMED,ConstraintStrength.HARD

def enrich_plan(plan: QueryPlan, adapter: DomainAdapter) -> QueryPlan:
    text = plan.original_text
    entities = []
    occupied = []
    for kind in ('instrument_model', 'software_version', 'numeric_unit', 'identifier_candidate'):
        pattern = adapter.patterns.get(kind, IDENTIFIER if kind=='identifier_candidate' else '')
        if not pattern: continue
        for m in re.finditer(pattern, text):
            if any(m.start()<b and m.end()>a for a,b in occupied): continue
            occupied.append(m.span())
            entity_type = kind
            reason = 'Explicit surface form matches configured pattern' if kind!='identifier_candidate' else 'Unregistered identifier retained verbatim; no inferred meaning'
            confidence = .95 if kind=='instrument_model' else .85 if kind!='identifier_candidate' else .65
            if kind=='identifier_candidate' and (
                re.search(r'(?:报|报警|错误码|故障码)\s*$', text[max(0,m.start()-8):m.start()])
                or re.match(r'\s*(?:报警|错误码|故障码)', text[m.end():])):
                entity_type, confidence = 'alarm_code', .95
                reason = 'Question explicitly associates this identifier with an alarm; code meaning is unknown'
            entities.append(QueryEntity(m.group(),entity_type,m.group().upper(),confidence,
                                        'domain_pattern' if kind in adapter.patterns else 'generic_pattern',
                                        'soft' if entity_type in ('identifier_candidate','numeric_unit') else 'explicit',reason))
    for m in re.finditer(MIXED_IDENTIFIER, text):
        if any(m.start()<b and m.end()>a for a,b in occupied): continue
        entities.append(QueryEntity(m.group(),'identifier_candidate',m.group().upper(),.65,'generic_pattern','soft','Unknown mixed identifier; not assumed to be an instrument model'))
    constraints = list(plan.constraints)
    for e in entities:
        if e.type == 'instrument_model':
            start=text.index(e.text)
            polarity,strength=surface_status(text,start,start+len(e.text))
            constraints.append(QueryConstraint(ConstraintKind.MODEL,e.normalized_value,strength==ConstraintStrength.HARD,e.text,e.confidence,
                polarity,strength,e.source,'Explicit model surface; original negation/uncertainty retained'))
    for m in re.finditer(r'(?:物料编码|物料号|part\s*(?:number|no\.?))\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9._-]{2,40})',text,re.I):
        entities.append(QueryEntity(m.group(1),'material_code',m.group(1).upper(),.95,'explicit_label','explicit','Explicitly labelled part/material identifier; no catalogue lookup'))
    for kind, groups in adapter.aliases.items():
        for value, words in groups.items():
            for word in words:
                pattern=re.escape(word)
                if word.isascii(): pattern=r'(?<![A-Za-z0-9_])'+pattern+r'(?![A-Za-z0-9_])'
                m=re.search(pattern,text,re.I)
                if not m: continue
                polarity,strength=surface_status(text,m.start(),m.end())
                uncertain=strength!=ConstraintStrength.HARD
                entity=QueryEntity(m.group(),kind,value,.75 if uncertain else .95,'domain_alias','soft' if uncertain else 'explicit','Explicit alias in original question; uncertainty retained' if uncertain else 'Explicit alias in original question')
                if not any(e.type==kind and e.normalized_value==value for e in entities): entities.append(entity)
                if kind in ('stage','scope','direction') and not any(c.kind.value==kind and c.value==value for c in constraints):
                    constraints.append(QueryConstraint(ConstraintKind(kind),value,not uncertain,m.group(),entity.confidence,
                        polarity,strength,
                        'domain_alias',entity.reason))
                break
    from local_full_text_search.rag.query_semantics import quantity_scopes
    for match, value in quantity_scopes(text):
        polarity, strength = surface_status(text, match.start(), match.end())
        if not any(c.kind == ConstraintKind.SCOPE and c.value == value for c in constraints):
            constraints.append(QueryConstraint(ConstraintKind.SCOPE, value, strength == ConstraintStrength.HARD,
                match.group(), .9, polarity, strength, 'quantity_span', 'Explicit quantifier and noun phrase; not tied to an alarm or assay'))
    metric_texts={e.text.casefold() for e in entities if e.type=='metric'}
    entities=[e for e in entities if not (e.type=='identifier_candidate' and e.text.casefold() in metric_texts)]
    # Multiple branch values must never become simultaneous global hard filters.
    for kind in (ConstraintKind.STAGE,ConstraintKind.SCOPE,ConstraintKind.MODEL,ConstraintKind.DIRECTION):
        values={c.value for c in constraints if c.kind==kind and c.polarity==ConstraintPolarity.AFFIRMED}
        if kind==ConstraintKind.DIRECTION: values.discard('abnormal')
        if len(values)>1:
            constraints=[replace(c,required=False,strength=ConstraintStrength.AMBIGUOUS,reason='Multiple values belong to unresolved branches; preserve all') if c.kind==kind else c for c in constraints]
    for c in constraints:
        if c.source_text and not any(e.type==c.kind.value and e.normalized_value==c.value for e in entities):
            entities.append(QueryEntity(c.source_text,c.kind.value,c.value,c.confidence or .85,c.source,c.strength.value,c.reason))
    requirements=[signal(name,f'Question contains “{m.group()}”') for name,pattern in REQUIREMENTS.items() if (m:=re.search(pattern,text))]
    if any(e.type=='alarm_code' for e in entities) and any(r.name=='handling' for r in requirements) and not any(r.name=='cause' for r in requirements):
        requirements.append(signal('cause','Diagnostic handling benefits from the documented trigger; inferred goal, not an asserted cause',.65))
    unknown=tuple(e for e in entities if e.type in ('identifier_candidate','alarm_code'))
    # Quoted user terminology is retained even outside the adapter vocabulary.
    for m in re.finditer(r'[“「"]([^”」"]{2,40})[”」"]',text):
        if not any(e.text==m.group(1) for e in entities):
            e=QueryEntity(m.group(1),'unknown_entity',m.group(1),.6,'question_quote','soft','User quoted term retained; no hard filtering')
            entities.append(e); unknown+= (e,)
    parts=[s.strip() for s in re.split(r'[？?；;]|[，,](?=如果|若|而|但|另一|分别)',text) if s.strip()]
    explicit_targets=sum(any(re.search(pattern,part) for pattern in REQUIREMENTS.values()) for part in parts)
    multi=len(requirements)>1 or explicit_targets>1 or bool(re.search(r'分别|区别|对比|如果|若',text))
    sub=[]
    if multi and len(parts)>1:
        for focus in parts[:4]:
            if len(focus)<4: continue
            sub.append(SubQuery(text+'\n关注问项：'+focus,focus,tuple(constraints),.9,'question_span',
                                'Explicit answer target/branch; full original question preserves all objects, models, stages and scopes'))
    # Activation is separate from extraction: low-confidence unknown terms remain
    # observable, but cannot change routing, filtering or generation instructions.
    strong_kinds={'alarm_code','instrument_model','material_code','product_line'}
    threshold=adapter.policy.get('activation_confidence',.9)
    anchors=[e for e in entities if e.type in strong_kinds and e.confidence>=threshold
             and surface_status(text,text.index(e.text),text.index(e.text)+len(e.text))[1]==ConstraintStrength.HARD]
    technical={e.type for e in entities if e.confidence>=threshold}
    active=adapter.identity!='generic' and bool(anchors or
        ('component' in technical and bool(technical & {'reagent','calibrator','stage','action','symptom'})) or
        ('stage' in technical and bool(technical & {'reagent','calibrator'})))
    scope_types=set(adapter.policy.get('scope_entity_types',[]))
    entities=[replace(e,is_strong_constraint=any(c.is_hard and c.source_text==e.text for c in constraints),
                      needs_scope_check=e.type in scope_types,
                      strength='soft' if surface_status(text,text.index(e.text),text.index(e.text)+len(e.text))[1]!=ConstraintStrength.HARD else e.strength)
              for e in entities if e.text in text]
    scope_entities=[e for e in entities if e.needs_scope_check]
    for kind,pattern in adapter.policy.get('scope_patterns',{}).items():
        for m in re.finditer(pattern,text):
            scope_entities.append(QueryEntity(m.group(),kind,m.group(),.85,'question_span','soft',
                'Explicit condition retained verbatim; no inferred applicability or hard exclusion',False,True))
    method=bool(re.search(r'如何判断|怎么判断|怎么排查|如何排查|什么时候|什么情况下|允许范围|允许误差|先后|步骤|流程|如果|若|仍然|(?:上次|前次).*(?:本次|当前)|(?:本次|当前).*(?:上次|前次)',text))
    requirement_names={s.name for s in requirements}
    # A direct troubleshooting/handling request is a method task even without
    # explicit "if/then" wording. Definition+handling still benefits from rows.
    method = method or ('definition' not in requirement_names and bool(
        requirement_names & {'handling','procedure','cause'}))
    strategies=[]
    if method: strategies.append(signal('method','Question asks for conditions, decisions or an ordered process'))
    elif any(e.type=='alarm_code' for e in entities): strategies.append(signal('alarm_explanation','Explicit alarm with answer goals'))
    if any(e.type=='numeric_unit' for e in entities) or re.search(r'多少|数值|阈值|计算|允许范围',text):
        strategies.append(signal('numeric','Preserve units and conditions; calculate only if requested and supported'))
    if scope_entities: strategies.append(signal('scoped','Keep source applicability local, never infer universal applicability'))
    branches=tuple(signal('preserve_branch',part,.9,'question_span') for part in parts) if method else ()
    ambiguities=[]
    if len({e.normalized_value for e in entities if e.type=='instrument_model'})>1:
        ambiguities.append(signal('multiple_models','Multiple models in question; bind each answer to its model, not a joint hard filter'))
    from local_full_text_search.rag.query_semantics import object_mentions
    mentions = object_mentions(text, adapter.aliases)
    if len({m['head']['normalized_value'] for m in mentions}) > 1:
        ambiguities.append(signal('part_whole', 'Multiple separately mentioned objects; bind each requested field to its own head noun, never substitute parent and child', .8))
    identifiers=[e for e in entities if e.type in ('alarm_code','instrument_model','material_code') and e.confidence>=threshold]
    hints=[]
    if identifiers:
        hints.extend(signal(name,'Explicit identifiers retained; exact text is a retrieval signal, not proof of applicability') for name in ('exact_identifier_match','protect_exact_evidence'))
    if any(e.type=='alarm_code' for e in entities) and not method: hints.append(signal('prefer_structured_row','Alarm explanation/handling may be recorded in keyed rows',.85))
    if len(requirements)>1: hints.append(signal('answer_slot_coverage','Multiple answer goals require separate evidence coverage'))
    terms=tuple(dict.fromkeys([e.text for e in identifiers]+[e.text for e in unknown]+list(plan.lexical_terms)))
    expression=' OR '.join('"'+t.replace('"','""')+'"' for t in terms)
    diagnostics=(signal('domain_config',adapter.load_reason,source='configuration'),
        signal('adapter_active' if active else 'adapter_inactive',
               'High-confidence explicit technical context' if active else 'No validated high-confidence domain context; original runtime query retained'))
    if not active:
        return replace(plan,entity_details=tuple(entities),unknown_entities=unknown,
            answer_requirements=tuple(requirements),scope_constraints=tuple(scope_entities),
            strategies=tuple(strategies),ambiguities=tuple(ambiguities),
            domain_config=adapter.identity,plan_diagnostics=diagnostics)
    return replace(plan,entity_details=tuple(entities),unknown_entities=unknown,
        entities=tuple(dict.fromkeys([*plan.entities,*[e.text for e in entities]])),
        answer_requirements=tuple(requirements),sub_queries=tuple(sub),retrieval_hints=tuple(hints),
        constraints=tuple(constraints),lexical_terms=terms,lexical_expression=expression,
        fallback_terms=tuple(dict.fromkeys([*terms,*plan.fallback_terms])),
        domain_config=adapter.identity,analyzer_version='query-plan-v5',adapter_active=True,
        strategies=tuple(strategies),scope_constraints=tuple(scope_entities),method_branches=branches,
        ambiguities=tuple(ambiguities),adapter_policy={**adapter.policy,'patterns':adapter.patterns,'aliases':adapter.aliases},
        protected_evidence=tuple(signal(name,'Evidence-layer requirement only; not a claim that evidence exists') for name in
            (('preserve_conditions','preserve_steps','bounded_exact') if method else ('bounded_exact','preserve_scope'))),
        plan_diagnostics=diagnostics)
