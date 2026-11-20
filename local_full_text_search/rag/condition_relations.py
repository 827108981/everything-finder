"""Conservative, source-bound applicability relations; no ranking or Gold access.

Only a local fact with an identifiable subject can prove an incompatibility.
Unsupported grammar remains UNKNOWN. This is an opt-in comparison strategy.
"""
from __future__ import annotations

import re
from .models import ConstraintJudgement, ConstraintKind as K, ConstraintStatus as S

_PARTS = re.compile(r'[^。；;！？?\n]+')
_MODEL = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]{1,6}-\d{3,6}[A-Za-z]?(?![A-Za-z0-9])')
_NEGATION = re.compile(r'(?:不是|并非|不属于|没有|未出现|无)\s*$')
_UNCERTAIN = re.compile(r'可能|疑似|好像|似乎|尚不确定')
_ACTION = re.compile(r'(?:重新|再次|执行|进行|完成|更换|检查|查看|选择|勾选|建议|需要|先|后)\s*$')
_BRANCH = re.compile(r'如果|假如|若|分别|否则|不同条件|(?:时|情况下)[，,:：]')


def _segments(text):
    return list(_PARTS.finditer(text))


def _affirmed(text, start):
    prefix = re.split(r'[，,。；;\n：:]', text[:start])[-1][-20:]
    quoted = any(m.start() <= start < m.end() for m in re.finditer(r'“[^”]*”|"[^"]*"|‘[^’]*’', text))
    return not quoted and not _NEGATION.search(prefix) and not _UNCERTAIN.search(prefix)


def _subject(prefix):
    # Strip grammar, not business terminology. Different noun phrases stay distinct.
    value = re.split(r'[，,。；;\n：:]', prefix)[-1].strip()
    value = re.sub(r'^(?:请问|现在|目前|只有|仅|所有|全部|多个|单个|大部分|大多数|某个|一个|这些|该|本次)+', '', value)
    value = re.sub(r'(?:均|都|普遍|明显|一直|持续|的结果|结果)$', '', value)
    return value.strip(' 的')


def _same_subject(a, b):
    if not a or not b:
        return False
    return a == b


def judge_relation(c, body, question, lexical_terms=()):
    """Return a judgement and auditable exact-body proof (offsets are real).

    No filename is an assertion about the ownership of every body fact.
    """
    proof = dict(condition_kind=c.kind.value, question_span=c.source_text,
                 status='unknown', candidate_quote='', source_start=None,
                 source_end=None, relation_basis='无法绑定', reason_code='binding_unknown')

    def result(status, reason, span=None, basis='局部正文', code=None):
        quote = body[span[0]:span[1]] if span else ''
        proof.update(status=status.value, candidate_quote=quote,
                     source_start=span[0] if span else None, source_end=span[1] if span else None,
                     relation_basis=basis if span else '无法绑定',
                     reason_code=code or {S.MATCH:'same_fact_supported', S.CONFLICT:'same_fact_conflict', S.UNKNOWN:'binding_unknown'}[status],
                     reason=reason)
        return ConstraintJudgement(c, status, reason, quote), proof

    if not c.is_hard:
        return result(S.UNKNOWN, '否定、猜测或未绑定的分支条件不能授权硬排除')
    if c.kind in {K.OBJECT, K.DEVICE, K.TOPIC}:
        for m in re.finditer(re.escape(c.value), body, flags=re.I):
            if _affirmed(body, m.start()):
                return result(S.MATCH, '正文直接出现目标对象；此判断不证明其他事实或全局范围', m.span(), code='body_object_mention')
        return result(S.UNKNOWN, '正文未确认目标对象；文件名不能代替事实归属')
    if c.kind is K.MODEL:
        target = (c.source_text or c.value).casefold()
        target_hits = [m for m in _MODEL.finditer(body) if m.group().casefold() == target and _affirmed(body, m.start())]
        if target_hits:
            return result(S.MATCH, '正文保留目标型号分支；不代表其他型号分支也适用', target_hits[0].span())
        # A scoped assertion, not a mention such as "与B型号比较".
        scoped = list(re.finditer(r'(?:仅适用于|适用型号\s*[:：]|型号\s*[:：])\s*('+_MODEL.pattern+r')', body))
        if len({m.group(1).casefold() for m in scoped}) == 1 and scoped and _affirmed(body, scoped[0].start()):
            return result(S.CONFLICT, '正文明确型号归属与目标型号不同', scoped[0].span())
        return result(S.UNKNOWN, '没有可靠的排他型号归属；另一型号的提及不足以证明冲突')
    if c.kind is K.SCOPE:
        from .condition_binding import scope_mentions, same_subject, disjoint_scope
        qhits = [x for x in scope_mentions(question) if x.source_text in c.source_text or c.source_text in x.source_text]
        if len(qhits) != 1 or qhits[0].role != 'observed' or _BRANCH.search(question):
            return result(S.UNKNOWN, '数量未绑定唯一已发生现象，或问题本身要求条件分支')
        q = qhits[0]
        hits = [x for x in scope_mentions(body) if x.role == 'observed' and x.is_hard
                and same_subject(q.subject, x.subject)]
        matched = next((x for x in hits if x.value == q.value), None)
        if matched:
            return result(S.MATCH, '同一对象的现象范围一致，其他方法分支仍完整保留', (matched.source_start, matched.source_end))
        # Whole multi-branch passages cannot be reduced to one global scene.
        from .condition_binding import SCOPE, LANG, WORDS
        all_values = {WORDS[m.group(1)] for m in SCOPE.finditer(body)}
        # Abbreviated quantifiers (单/多 + a counted noun) are still branches.
        # They must not disappear merely because a long-form lexicon missed them.
        short_quantities = re.compile(r'(单|多)(?:个)?(?:'+'|'.join(map(re.escape, LANG['counted_heads']))+r')')
        all_values.update({'单':'single', '多':'multiple'}[m.group(1)] for m in short_quantities.finditer(body))
        if len(all_values) > 1 or _BRANCH.search(body):
            return result(S.UNKNOWN, '正文有多个范围或条件分支，不以单一局部量词淘汰整段')
        if len(hits) == 1 and disjoint_scope(q.value, hits[0].value):
            h = hits[0]
            return result(S.CONFLICT, '同一对象唯一现象的数量范围明确不相容', (h.source_start, h.source_end))
        return result(S.UNKNOWN, '范围未绑定同一故障对象，或取值可能重叠')
    if c.kind in {K.STAGE, K.DIRECTION}:
        from .constraint_matcher import _VALUES
        words = sorted(((word, value) for value, phrases in _VALUES[c.kind].items() for word in phrases), key=lambda p:-len(p[0]))
        hits, occupied = [], []
        qstart = question.find(c.source_text) if c.source_text else -1
        qsubject = c.subject or (_subject(question[:qstart]) if qstart >= 0 else '')
        for word, value in words:
            for m in re.finditer(re.escape(word), body):
                if any(a < m.end() and m.start() < b for a,b in occupied):
                    continue
                occupied.append(m.span())
                if not _affirmed(body, m.start()):
                    continue
                segment = next(s for s in _segments(body) if s.start() <= m.start() < s.end())
                before, after = body[segment.start():m.start()], body[m.end():segment.end()]
                if c.kind is K.STAGE:
                    if word in {'样本', '标本', '质控品'} or _ACTION.search(before):
                        continue
                    # A stage needs an explicit temporal/scene predicate, not a material noun.
                    if not re.match(r'(?:过程|阶段|期间|时|中|失败|异常|报错|报警|不通过|结果)', after.strip()):
                        continue
                elif not _same_subject(qsubject, _subject(before)):
                    continue
                hits.append((value, segment.span()))
        matched = next((span for value,span in hits if value == c.value or
                        c.kind is K.DIRECTION and c.value == 'abnormal' and value in {'low','high'}), None)
        if matched:
            return result(S.MATCH, '局部事实确认目标条件；不将其他处理动作或分支泛化', matched)
        values = {value for value,_ in hits}
        if c.kind is K.DIRECTION and c.value in {'low','high'} and len(values)==1 and values <= {'low','high'} and not _BRANCH.search(body):
            return result(S.CONFLICT, '同一属性、同一对象的唯一方向明确相反', hits[0][1])
        if c.kind is K.STAGE:
            # Only an explicit scene declaration can authorize a stage exclusion.
            declarations = [s for s in _segments(body) if re.match(r'\s*(?:故障阶段|异常阶段|发生阶段|检测阶段)\s*[:：]', s.group())]
            incompatible = [span for value,span in hits if value != c.value and any(s.span()==span for s in declarations)]
            if incompatible and len(values)==1 and not _BRANCH.search(body):
                return result(S.CONFLICT, '正文明确声明唯一故障阶段，与问题阶段不相容', incompatible[0])
        return result(S.UNKNOWN, '条件未绑定同一事实，或其他取值可能是处理步骤和方法分支')
    return result(S.UNKNOWN, '该关系类型尚无可验证的局部判定能力')
