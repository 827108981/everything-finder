"""Question-only semantics. No corpus, answer catalogue, or evaluation imports."""
from dataclasses import asdict
import re


FIELD_PATTERNS = {
    'definition': r'什么意思|是什么|含义|定义|代表什么',
    'cause': r'原因|为什么|怎么回事|什么导致',
    'solution': r'怎么处理|如何处理|怎么办|怎么解决|如何解决|排查|检查什么|检查哪些',
    'procedure': r'流程|步骤|先后|顺序|如何操作|怎么操作',
    'comparison': r'比较|区别|对比|分别',
    'condition': r'如何判断|怎么判断|什么时候|什么情况下|是否超过',
    'lifetime': r'寿命|使用次数|用多少次|更换周期',
    'material_code': r'物料编码|物料号|备件号|配件编码',
    'value': r'多少|数值|电压|电流|流量|温度|压力|规格|参数',
    'confirmation': r'能否.*(?:确定|判定|确认)|能.*直接.*(?:判断|确认)|是否.*(?:损坏|故障)',
    'replacement': r'更换哪|换哪个|换什么|更换件|怎么更换|如何更换',
    'enumeration': r'哪些|哪几|种类|类型|支持什么',
}


def quantity_scopes(text):
    """Quantity + bounded noun phrase; does not enumerate a particular assay."""
    pattern = r'(单个|单一|单|一个|个别|少数|几个|多个|多|部分|大部分|大多数|全部|所有|每个)([^，。；？\n]{0,8}?)(项目|通道|模块|样本|设备|仪器)'
    values = {'单个': 'single', '单一': 'single', '单': 'single', '一个': 'single', '个别': 'single',
              '全部': 'all', '所有': 'all', '每个': 'all'}
    return [(m, values.get(m[1], 'multiple')) for m in re.finditer(pattern, text)]


def object_mentions(text, aliases):
    """Bind adjacent noun mentions to a compound; never invent a parts catalogue."""
    hits = []
    for kind, groups in aliases.items():
        if kind not in {'system', 'module', 'component', 'subcomponent', 'consumable', 'reagent', 'calibrator'}:
            continue
        for normalized, words in groups.items():
            for word in words:
                for m in re.finditer(re.escape(word), text):
                    hits.append({'text': m.group(), 'normalized_value': normalized, 'type': kind,
                                 'start': m.start(), 'end': m.end()})
    hits.sort(key=lambda r: (r['start'], -r['end']))
    maximal = [r for r in hits if not any(o is not r and o['start'] <= r['start'] and o['end'] >= r['end'] and o['end']-o['start'] > r['end']-r['start'] for o in hits)]
    groups = []
    for hit in maximal:
        if groups and hit['start'] >= groups[-1][-1]['end'] and re.fullmatch(r'\s*(?:的)?\s*', text[groups[-1][-1]['end']:hit['start']]):
            groups[-1].append(hit)
        else:
            groups.append([hit])
    result = []
    for group in groups:
        head = group[-1]
        result.append({'text': text[group[0]['start']:head['end']], 'head': head,
                       'modifiers': group[:-1], 'confidence': .9 if len(group) > 1 else .85,
                       'source': 'question_spans', 'reason': 'Adjacent noun phrase; rightmost noun is the head, no inferred replacement relationship'})
    return result


def query_spec(plan):
    fields = [name for name, pattern in FIELD_PATTERNS.items() if re.search(pattern, plan.original_text)]
    entities = {}
    for entity in plan.entity_details:
        entities.setdefault(entity.type, []).append(asdict(entity))
    scope = {}
    for entity in plan.scope_constraints:
        scope.setdefault(entity.type, []).append(asdict(entity))
    ambiguous = [asdict(s) for s in plan.ambiguities]
    # This is a request mode, not an answerability decision made without evidence.
    conditional = bool(re.search(r'如果|若|分别|什么情况下|什么时候', plan.original_text))
    return {'intent': fields or [plan.intent], 'entities': entities, 'scope': scope,
            'constraints': [asdict(c) for c in plan.constraints], 'requested_fields': fields,
            'ambiguity': ambiguous, 'answer_mode': 'conditional' if conditional else 'answer',
            'object_mentions': object_mentions(plan.original_text, plan.adapter_policy.get('aliases', {})),
            'query_variants': [{'text': plan.original_text, 'source': 'original', 'confidence': 1., 'reason': 'Original question retained'},
                *[{'text': v.query_text, 'source': v.route_id, 'confidence': .8, 'reason': 'Retrieval expression only, not additional facts'} for v in plan.semantic_variants if v.query_text != plan.original_text]],
            'mode_reason': 'Question intent only; evidence sufficiency is evaluated downstream'}
