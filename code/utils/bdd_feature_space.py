import json
import math
import os
import random
from collections import OrderedDict
from functools import lru_cache

from dependency.dependency_manager import DependencyManager
from utils.knob_space_utils import build_categorical_values, canonicalize_value


class BDDFeatureSpace:
    """A small ordered decision diagram over discretized knob-value features.

    This is an in-repo MVP for BDD-style legal-space sampling without external
    BDD dependencies. Each knob is encoded as exactly one selected value feature.
    Counts are memoized over the ordered variables, giving uniform sampling over
    all discrete configurations satisfying static dependency rules.
    """

    def __init__(self, knobs_info, rule_file="", numeric_bins=5, seed=0):
        self.knobs_info = knobs_info
        self.rules = self._load_rules(rule_file)
        self.numeric_bins = max(3, int(numeric_bins))
        self.rng = random.Random(int(seed))
        self.domains = self._build_domains()
        self.names = list(self.domains.keys())
        self.rule_knobs = self._collect_rule_knobs()
        self.feature_names = self._build_feature_names()

    def sample(self, fixed=None):
        fixed = self._normalize_fixed(fixed or {})
        total = self.count(fixed)
        if total <= 0:
            return None

        partial = {}
        for idx, name in enumerate(self.names):
            values = [fixed[name]] if name in fixed else self.domains[name]
            weighted = []
            for value in values:
                next_partial = dict(partial)
                next_partial[name] = value
                if not self._partial_valid(next_partial):
                    continue
                cnt = self._count_from(idx + 1, self._state_tuple(next_partial, fixed))
                if cnt > 0:
                    weighted.append((value, cnt))
            if not weighted:
                return None
            pick = self.rng.randrange(sum(cnt for _, cnt in weighted))
            upto = 0
            for value, cnt in weighted:
                upto += cnt
                if pick < upto:
                    partial[name] = value
                    break
        return partial if self._full_valid(partial) else None

    def count(self, fixed=None):
        fixed = self._normalize_fixed(fixed or {})
        return self._count_from(0, self._state_tuple({}, fixed))

    def common_features(self, configs):
        if not configs:
            return {}
        common = {}
        for name in self.names:
            first = configs[0].get(name)
            if all(self._value_key(config.get(name)) == self._value_key(first) for config in configs[1:]):
                common[name] = first
        return common

    def feature_summary(self):
        return {
            "knobs": len(self.domains),
            "boolean_features": sum(len(values) for values in self.domains.values()),
            "legal_configurations": self.count(),
            "rules": len(self.rules),
        }

    def _count_from(self, idx, state):
        fixed, partial = self._state_to_dicts(state)
        return self._cached_count(idx, self._dict_tuple(fixed), self._dict_tuple(partial))

    @lru_cache(maxsize=None)
    def _cached_count(self, idx, fixed_tuple, partial_tuple):
        fixed = dict(fixed_tuple)
        partial = dict(partial_tuple)
        if not self._partial_valid(partial):
            return 0
        if idx >= len(self.names):
            return 1 if self._full_valid(partial) else 0

        name = self.names[idx]
        values = [fixed[name]] if name in fixed else self.domains[name]
        total = 0
        for value in values:
            next_partial = dict(partial)
            next_partial[name] = value
            if not self._partial_valid(next_partial):
                continue
            total += self._cached_count(
                idx + 1,
                fixed_tuple,
                self._dict_tuple(self._relevant_partial(next_partial, fixed)),
            )
        return total

    def _partial_valid(self, partial):
        for rule in self.rules:
            if self._rule_definitely_violated(rule, partial):
                return False
        return True

    def _full_valid(self, config):
        return not any(self._rule_violated(rule, config) for rule in self.rules)

    def _rule_definitely_violated(self, rule, partial):
        if rule.get("type", "control") != "control":
            return False
        rule_knobs = set(rule.get("if", {})) | set(rule.get("then", {}))
        if not rule_knobs.issubset(partial):
            return False
        return self._rule_violated(rule, partial)

    def _rule_violated(self, rule, config):
        if rule.get("type", "control") != "control":
            return False

        for key, expected in rule.get("if", {}).items():
            if key not in config or not DependencyManager._matches(config[key], expected, config):
                return False

        for key, allowed in rule.get("then", {}).items():
            allowed_values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            if key not in config:
                return True
            if not any(DependencyManager._matches(config[key], candidate, config) for candidate in allowed_values):
                return True
        return False

    def _build_domains(self):
        domains = OrderedDict()
        thresholds = self._dependency_thresholds()
        for name, info in self.knobs_info.items():
            ktype = str(info.get("type", "")).lower()
            if ktype in ("enum", "boolean"):
                values = build_categorical_values(info)
            elif ktype in ("integer", "int", "float", "num"):
                values = self._numeric_values(name, info, thresholds.get(name, []))
            else:
                values = build_categorical_values(info)
            domains[name] = self._dedupe_values(info, values)
        return domains

    def _numeric_values(self, name, info, extra):
        ktype = str(info.get("type", "")).lower()
        caster = int if ktype in ("integer", "int") else float
        lower = self._numeric_or_none(info.get("min"))
        upper = self._numeric_or_none(info.get("max"))
        default = self._numeric_or_none(info.get("default"))
        if lower is None and upper is None:
            values = [info.get("default")]
            return [v for v in values if v is not None]
        if lower is None:
            lower = default if default is not None else upper
        if upper is None:
            upper = default if default is not None else lower
        if lower > upper:
            lower, upper = upper, lower

        values = [lower, upper]
        if default is not None:
            values.append(default)
        if self.numeric_bins > 3 and upper != lower:
            for i in range(1, self.numeric_bins - 1):
                ratio = i / (self.numeric_bins - 1)
                values.append(lower + ratio * (upper - lower))
        values.extend(v for v in extra if v is not None)

        normalized = []
        for value in values:
            try:
                value = caster(value)
            except Exception:
                continue
            if value < lower:
                value = caster(lower)
            if value > upper:
                value = caster(upper)
            normalized.append(value)
        return normalized

    def _dependency_thresholds(self):
        thresholds = {}
        for rule in self.rules:
            for conds in (rule.get("if", {}), rule.get("then", {})):
                for knob, expected in conds.items():
                    thresholds.setdefault(knob, [])
                    for value in self._condition_values(expected):
                        thresholds[knob].append(value)
        return thresholds

    def _condition_values(self, expected):
        if isinstance(expected, dict):
            values = []
            for value in expected.values():
                values.extend(self._condition_values(value))
            return values
        if isinstance(expected, (list, tuple, set)):
            values = []
            for item in expected:
                values.extend(self._condition_values(item))
            return values
        return [self._numeric_or_none(expected)]

    def _dedupe_values(self, info, values):
        result = []
        seen = set()
        for value in values:
            if value is None:
                continue
            value = canonicalize_value(info, value, for_optimizer=False)
            key = self._value_key(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        if not result and info.get("default") is not None:
            result.append(info.get("default"))
        return result

    def _normalize_fixed(self, fixed):
        normalized = {}
        for name, value in fixed.items():
            if name not in self.domains:
                continue
            info = self.knobs_info.get(name, {})
            value = canonicalize_value(info, value, for_optimizer=False)
            domain_by_key = {self._value_key(v): v for v in self.domains[name]}
            key = self._value_key(value)
            if key in domain_by_key:
                normalized[name] = domain_by_key[key]
        return normalized

    def _collect_rule_knobs(self):
        knobs = set()
        for rule in self.rules:
            knobs.update(rule.get("if", {}).keys())
            knobs.update(rule.get("then", {}).keys())
        return knobs

    def _relevant_partial(self, partial, fixed):
        relevant = self.rule_knobs | set(fixed)
        return {k: v for k, v in partial.items() if k in relevant}

    def _state_tuple(self, partial, fixed):
        return (self._dict_tuple(fixed), self._dict_tuple(self._relevant_partial(partial, fixed)))

    def _state_to_dicts(self, state):
        fixed_tuple, partial_tuple = state
        return dict(fixed_tuple), dict(partial_tuple)

    def _dict_tuple(self, values):
        return tuple(sorted((k, self._hashable(v)) for k, v in values.items()))

    def _build_feature_names(self):
        features = {}
        for knob, values in self.domains.items():
            features[knob] = [f"{knob}=={value}" for value in values]
        return features

    @staticmethod
    def _load_rules(rule_file):
        if not rule_file or not os.path.exists(rule_file):
            return []
        with open(rule_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rules = raw.get("rules", []) if isinstance(raw, dict) else raw
        normalized = []
        for idx, rule in enumerate(rules or [], 1):
            if not isinstance(rule, dict):
                continue
            if not isinstance(rule.get("if", {}), dict) or not isinstance(rule.get("then", {}), dict):
                continue
            normalized.append(
                {
                    "id": rule.get("id", f"rule_{idx}"),
                    "type": rule.get("type", "control"),
                    "if": rule.get("if", {}),
                    "then": rule.get("then", {}),
                }
            )
        return normalized

    @staticmethod
    def _numeric_or_none(value):
        try:
            if isinstance(value, str) and value.strip().lower() == "unset":
                return None
            num = float(value)
            if math.isnan(num) or math.isinf(num):
                return None
            return num
        except Exception:
            return None

    @staticmethod
    def _hashable(value):
        if isinstance(value, float):
            return round(value, 12)
        return value

    @staticmethod
    def _value_key(value):
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, float):
            return round(value, 12)
        return value
