"""Conservative extractive facts, value types and explicit source-local scopes."""
from dataclasses import asdict, dataclass, field as dataclass_field
from hashlib import sha256
import json
import re


MODEL = r'(?<![A-Za-z0-9])[A-Za-z]{1,6}-[0-9]{3,6}[A-Za-z]?(?![A-Za-z0-9])'
NUMBER = r'(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:\s*(?:[～~±–—]|至|\.\.)\s*\d+(?:\.\d+)?)?\s*(?:[μµu]?L(?:/s)?|mL(?:/min)?|m?V|m?A|k?Pa|rpm|CPS|RLU|mmol/L|mm|℃|°C|%|次(?:测试)?)?'
VALUE_ROLES = {'nominal': r'额定|标称|nominal|rated', 'typical': r'典型|typical',
               'normal_range': r'正常.*范围|工作范围|允许范围', 'threshold': r'阈值|不超过|至少|上限|下限',
               'control_signal': r'控制信号|PWM|脉冲|占空比', 'measured': r'实测|测得|测量值'}


def normalized(text):
    return re.sub(r'\s+', '', text).casefold()


def explicit_scope(text):
    result = {}
    for line in text.splitlines():
        m = re.match(r'\s*(适用型号|适用范围|适用平台|产品线|软件版本|阶段|对象)\s*[:：]\s*(.+)', line)
        if m:
            result.setdefault(m[1], []).append(m[2].strip())
        elif re.search(r'仅适用|仅用于|只适用', line):
            result.setdefault('restriction', []).append(line.strip())
    return result


def numeric_semantics(statement):
    # Preserve context; no conversion between nominal, tolerance and observed values.
    out = []
    for m in re.finditer(NUMBER, statement, re.I):
        local = statement[max(0, m.start()-35):m.end()+20]
        roles = [role for role, pattern in VALUE_ROLES.items() if re.search(pattern, local, re.I)]
        out.append({'text': m.group().strip(), 'types': roles or ['unspecified'],
                    'object_text': statement[:m.start()].strip(), 'start': m.start(), 'end': m.end(),
                    'confidence': .8, 'source': 'indexed_text_span', 'reason': 'Explicit value and local labels; no inferred identity or unit conversion'})
    return out


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    statement: str
    fact_type: str
    scope: dict
    conditions: tuple[str, ...]
    object: dict
    source_ref: dict
    evidence_role: str
    confidence: float
    field: str = ''
    values: tuple[str, ...] = ()
    source_refs: tuple[dict, ...] = ()
    diagnostics: dict = dataclass_field(default_factory=dict)


def extract_facts(text, source, source_id='', role='primary'):
    """Every statement remains an exact original line. Untyped prose is unknown."""
    scope = explicit_scope(text)
    context_object = scope.get('对象', [])
    condition = ()
    kind = 'definition'
    facts = []
    headings = {'cause': r'可能原因|故障原因|原因分析|原因', 'solution': r'解决思路|解决方法|处理方法|处理步骤',
                'condition': r'前提|条件|注意事项', 'definition': r'含义|定义|描述'}
    for index, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        for name, pattern in headings.items():
            if re.match(r'^(?:' + pattern + r')\s*[:：]?$', line):
                kind = name
        if re.match(r'^(?:如果|若|当|仅当)', line):
            condition = (line,)
        pair = re.match(r'^([^：:\n]{2,30})\s*[:：]\s*(.+)', line)
        label, value = pair.groups() if pair else ('', '')
        numbers = numeric_semantics(line)
        fact_type = ('range' if any(re.search(r'～|~|±|至', n['text']) for n in numbers) else 'value') if numbers else kind
        ref = {'file': source.file_path, 'page': source.page_number or source.slide_number,
               'chunk_id': source.stable_chunk_id, 'location': source.location_text, 'source_id': source_id,
               'line': index+1}
        key = sha256((json.dumps(ref, sort_keys=True, ensure_ascii=False)+line).encode()).hexdigest()[:24]
        facts.append(asdict(EvidenceFact(key, line, fact_type, scope, condition,
            {'explicit': context_object, 'field_label': label, 'binding': 'explicit' if context_object else 'unknown'},
            ref, role, .8 if pair else .55, label, (value,) if value else (), (ref,),
            {'numeric_semantics': numbers, 'certainty': 'possible' if kind == 'cause' or '可能' in line else 'supported',
             'sequence_index': index, 'source': 'verbatim_line', 'reason': 'Exact text; inferred object bindings are not asserted'})))
    return facts


def merge_facts(facts):
    """Unknown object/scope facts are local; explicit equal keys permit agreement."""
    groups = {}
    for fact in facts:
        local = '' if fact['scope'] and fact['object'].get('explicit') else fact['source_ref']['chunk_id']
        key = json.dumps([fact['field'], fact['object'], fact['scope'], fact['conditions'], local], ensure_ascii=False, sort_keys=True)
        groups.setdefault(key, []).append(fact)
    result = []
    for rows in groups.values():
        distinct = {}
        for fact in rows:
            value = normalized(fact['statement'])
            if value not in distinct:
                distinct[value] = {**fact, 'source_refs': list(fact['source_refs'])}
            else:
                for ref in fact['source_refs']:
                    if ref not in distinct[value]['source_refs']:
                        distinct[value]['source_refs'].append(ref)
        if rows[0]['field'] and len({tuple(r['values']) for r in rows}) > 1:
            for fact in distinct.values():
                fact['evidence_role'] = 'conflict'
        result.extend(distinct.values())
    return result
