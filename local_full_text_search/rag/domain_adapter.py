"""Optional declarative vocabulary; no source locations, answers or alarm catalogue."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re


@dataclass(frozen=True)
class DomainAdapter:
    aliases: dict = field(default_factory=dict)
    patterns: dict = field(default_factory=dict)
    identity: str = "generic"
    load_reason: str = "No domain configuration; generic parsing retained"
    policy: dict = field(default_factory=dict)

    @classmethod
    def load(cls, config: str | Path | None = None):
        if not config:
            return cls()
        path = (Path(__file__).parent / 'domain_configs' / 'chemiluminescence.json'
                if str(config) == 'chemiluminescence' else Path(config))
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
            forbidden = {'question_id', 'expected_answer', 'fixed_page', 'gold_fact', 'special_case_for_xxx',
                         'required_facts', 'forbidden_claims', 'gold_evidence', 'hard_negatives', 'specific_component_names'}
            def check_keys(node):
                if isinstance(node, dict):
                    if any(k in forbidden or k.startswith('special_case_for_') for k in node):
                        raise ValueError('Evaluation data or case-specific configuration is prohibited')
                    for value in node.values(): check_keys(value)
                elif isinstance(node, list):
                    for value in node: check_keys(value)
            check_keys(data)
            aliases, patterns = data.get('aliases', {}), data.get('patterns', {})
            if not isinstance(aliases, dict) or not isinstance(patterns, dict):
                raise ValueError('aliases and patterns must be objects')
            for kind, values in aliases.items():
                if not isinstance(values, dict): raise ValueError('alias group must be an object')
                for name, words in values.items():
                    if not isinstance(words, list) or not all(isinstance(w, str) and w for w in words):
                        raise ValueError('aliases must be nonempty strings')
            for pattern in patterns.values(): re.compile(pattern)
            policy=data.get('policy', {})
            if not isinstance(policy, dict): raise ValueError('policy must be an object')
            threshold=policy.get('activation_confidence', .9)
            cap=policy.get('max_exact_representatives', 2)
            if not isinstance(threshold,(int,float)) or not .8 <= threshold <= 1: raise ValueError('invalid activation threshold')
            if type(cap) is not int or not 0 <= cap <= 3: raise ValueError('invalid representative cap')
            for pattern in policy.get('scope_patterns',{}).values(): re.compile(pattern)
            if not isinstance(policy.get('scope_entity_types',[]),list): raise ValueError('scope types must be a list')
            return cls(aliases, patterns, f'{path.name}:{hashlib.sha256(raw).hexdigest()[:16]}', 'Domain configuration loaded',policy)
        except (OSError, ValueError, TypeError, re.error) as exc:
            return cls(load_reason=f'Domain configuration unavailable; generic fallback: {type(exc).__name__}')
