"""Opt-in, source-bound applicability checks. No retrieval, Gold, or answer input.

Unknown syntax remains unknown. A local mention is not an exclusive document
scope; mixed branches are kept whole. Constructor switches permit isolated tests.
"""
from dataclasses import asdict, replace
from pathlib import Path
import json
import re

from .models import (CandidateDecision, ConstraintJudgement, ConstraintKind as K,
                     ConstraintPolarity as P, ConstraintStatus as S,
                     ConstraintStrength as T, QueryConstraint)

LANG = json.loads((Path(__file__).parent/'domain_configs/condition_language.json').read_text(encoding='utf8'))
WORDS = {w: v for v, words in LANG['quantifiers'].items() for w in words}
SCOPE = re.compile('('+'|'.join(map(re.escape, sorted(WORDS, key=len, reverse=True)))+
                   r')([\u3400-\u9fffA-Za-z0-9-]{0,8}?)('+'|'.join(LANG['counted_heads'])+')')
MODELS = re.compile(LANG['model_pattern'])
MANAGED = {K.SCOPE, K.STAGE, K.DIRECTION, K.MODEL}


def polarity(text, start, end):
    prefix = re.split(r'[，,。；;\n]', text[:start])[-1][-16:]
    if any(a.start() <= start < a.end() for a in re.finditer(r'“[^”]*”|"[^"]*"', text)):
        return P.UNKNOWN, T.AMBIGUOUS
    if re.search('(?:'+LANG['negation_pattern']+r')\s*$', prefix):
        return P.NEGATED, T.AMBIGUOUS
    if re.search(LANG['uncertain_pattern'], prefix):
        return P.AFFIRMED, T.SOFT
    return P.AFFIRMED, T.HARD


def scope_mentions(text):
    result = []
    for m in SCOPE.finditer(text):
        p, strength = polarity(text, *m.span())
        suffix = re.split(r'[。；;\n]', text[m.end():])[0][:35]
        # Instructions to inspect all items are not observations that all fail.
        prefix = re.split(r'[，,。；;\n]', text[:m.start()])[-1]
        observation = bool(re.search(LANG['symptom_pattern'], suffix)) and not re.search(r'(?:检查|查看|选择|勾选|检测)\s*$', prefix)
        result.append(QueryConstraint(K.SCOPE, WORDS[m[1]], strength is T.HARD and p is P.AFFIRMED,
            m.group(), .95, p, strength, 'bound_span', '数量词绑定紧邻对象；范围原文不改写',
            m[2]+m[3], m.start(), m.end(), 'observed' if observation else 'mention'))
    return result


def subject_before(text, start):
    prefix = re.split(r'[，,。；;\n：:]', text[:start])[-1]
    prefix = re.sub(r'(?:不是|并非|好像|有点|可能|均|都|普遍|的结果|结果)$', '', prefix)
    match = re.search(r'([A-Za-z0-9\u3400-\u9fff-]{1,16})$', prefix)
    return match.group(1) if match else ''


def bind_plan(plan):
    """Enrich applicability only. Lexical/vector expressions remain unchanged."""
    raw = plan.original_text
    scopes = scope_mentions(raw)
    conditional = bool(re.search(LANG['branch_pattern'], raw))
    constraints = [c for c in plan.constraints if c.kind is not K.SCOPE] if scopes else list(plan.constraints)
    for c in scopes:
        if c.role != 'observed' or conditional:
            c = replace(c, required=False, strength=T.AMBIGUOUS, role='branch' if conditional else c.role,
                        reason='分支/提问对象，不是唯一已发生故障条件')
        constraints.append(c)
    bound = []
    for c in constraints:
        if c.source == 'bound_span':
            bound.append(c); continue
        start = raw.find(c.source_text) if c.source_text else -1
        subject = subject_before(raw, start) if start >= 0 and c.kind is K.DIRECTION else ''
        cc = replace(c, subject=subject, source_start=start if start >= 0 else None,
                     source_end=start+len(c.source_text) if start >= 0 else None)
        if conditional and c.kind in MANAGED:
            cc = replace(cc, required=False, strength=T.AMBIGUOUS, role='branch')
        bound.append(cc)
    return replace(plan, constraints=tuple(bound))


def same_subject(a, b):
    if not a or not b: return False
    if a == b: return True
    # A generic noun may bind a modified noun; two different modifiers do not.
    return any((a == head and b.endswith(head)) or (b == head and a.endswith(head))
               for head in LANG['counted_heads'])


def disjoint_scope(a, b):
    return frozenset((a, b)) in {
        frozenset(('single', 'multiple')), frozenset(('single', 'most')),
        frozenset(('single', 'all')), frozenset(('few', 'most')), frozenset(('few', 'all'))}


def judge_local(c, body, question):
    unknown = lambda why: ConstraintJudgement(c, S.UNKNOWN, why)
    if not c.is_hard:
        return unknown('否定、不确定或方法分支不构成全局硬条件')
    if c.kind is K.SCOPE:
        q = next((x for x in scope_mentions(question) if x.source_text == c.source_text), None)
        if q is None: return unknown('无法定位数量与对象绑定')
        observed = [x for x in scope_mentions(body) if x.role == 'observed' and x.is_hard]
        related = [x for x in observed if same_subject(q.subject, x.subject)]
        # Multiple scenes, including a scene for another object, cannot be
        # compressed into a global hard rejection without structural ownership.
        if len({(x.subject, x.value) for x in observed}) > 1:
            if any(x.value == q.value for x in related):
                return ConstraintJudgement(c, S.MATCH, '含适用分支；保留整段，匹配不代表其他分支也适用',
                                           next(x.source_text for x in related if x.value == q.value))
            return unknown('多个对象或范围分支，不能整段淘汰')
        if not related: return unknown('正文未明确说明同一对象的故障范围')
        x = related[0]
        if x.value == q.value:
            return ConstraintJudgement(c, S.MATCH, '同一对象的故障范围一致', x.source_text)
        if disjoint_scope(q.value, x.value):
            return ConstraintJudgement(c, S.CONFLICT, '同一对象、唯一已发生场景的数量范围明确不相容', x.source_text)
        return unknown('数量范围可能重叠；不同取值不足以证明冲突')
    if c.kind in {K.STAGE, K.DIRECTION}:
        from .constraint_matcher import _VALUES
        phrases = sorted(((w,v) for v, ws in _VALUES[c.kind].items() for w in ws), key=lambda p: -len(p[0]))
        hits = []; occupied = []
        subject = c.subject or (subject_before(question, question.find(c.source_text)) if c.source_text in question else '')
        for word,value in phrases:
            for m in re.finditer(re.escape(word), body):
                if any(a < m.end() and m.start() < b for a,b in occupied): continue
                occupied.append(m.span())
                p,t = polarity(body,*m.span())
                if p is not P.AFFIRMED or t is not T.HARD: continue
                if c.kind is K.DIRECTION:
                    other = subject_before(body,m.start())
                    if not subject or subject != other: continue
                elif word in {'样本','标本','质控品'}:
                    continue  # Material nouns alone do not establish a testing stage.
                hits.append((value,m.group()))
        vals = {v for v,_ in hits}
        if c.value in vals:
            return ConstraintJudgement(c,S.MATCH,'局部正文支持；多个分支仍完整保留',next(w for v,w in hits if v==c.value))
        if c.kind is K.STAGE:
            return unknown('其他阶段可能是操作步骤，不能据此否定当前故障阶段')
        if len(vals)==1 and c.value in {'low','high'} and next(iter(vals)) in {'low','high'}:
            return ConstraintJudgement(c,S.CONFLICT,'同一对象的局部方向明确相反',hits[0][1])
        return unknown('同一对象方向未获局部确认，或涉及多个分支')
    if c.kind is K.MODEL:
        models = {m.group().casefold() for m in MODELS.finditer(body)}
        if c.value.casefold() in models:
            return ConstraintJudgement(c,S.MATCH,'正文出现用户型号；不扩展其他型号适用性',c.value)
        explicit = re.search(r'(?:仅适用于|仅支持)\s*('+LANG['model_pattern']+')',body)
        if explicit and len(models)==1:
            return ConstraintJudgement(c,S.CONFLICT,'正文明确排他型号限定',explicit.group())
        return unknown('文件名或其他型号的提及不足以证明排他适用范围')
    return None


def evaluate_local(plan, candidates, adapter):
    from .constraint_matcher import ConstraintMatcher
    from .query_interpreter import QueryInterpreter
    # Preserve the existing ordering; no additional condition bonus introduced.
    matcher = ConstraintMatcher(domain_adapter=adapter)
    ordering_plan = plan if plan.adapter_active else QueryInterpreter().interpret(plan.original_text)
    ranked = matcher.evaluate(ordering_plan, candidates)
    original_order = [x.chunk.stable_chunk_id for x in ranked]
    checked = matcher.evaluate(plan, ranked)
    by_id = {x.chunk.stable_chunk_id:x for x in checked}
    ranked = tuple(by_id[cid] for cid in original_order)
    for item in ranked:
        old = item.constraint_decision
        js = tuple(judge_local(j.constraint,item.chunk.text,plan.original_text)
                   if j.constraint.kind in MANAGED else j for j in old.judgements)
        hard = [j for j in js if j.constraint.is_hard]
        conflicts = [j for j in hard if j.status is S.CONFLICT]
        matches = [j for j in hard if j.status is S.MATCH]
        unknowns = [j for j in hard if j.status is S.UNKNOWN]
        item.constraint_decision = replace(old, accepted=not conflicts, judgements=js,
            required_match_count=len(matches), required_unknown_count=len(unknowns), required_conflict_count=len(conflicts),
            reason=('硬条件冲突：'+'；'.join(j.reason+'：'+j.matched_text for j in conflicts)) if conflicts
                   else '局部条件检查：未证明冲突；未知与多分支保留',
            support_status='conflict' if conflicts else 'unknown' if unknowns else 'supported',
            support_spans=tuple(j.matched_text for j in matches))
        item.query_plan_diagnostics['local_conditions'] = {
            'version':'local-binding-v1', 'order':'legacy condition order preserved; no new bonus',
            'checks':[{'condition':asdict(j.constraint),'status':j.status.value,'reason':j.reason,'quote':j.matched_text} for j in js]}
    return ranked
