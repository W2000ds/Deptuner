import json


class DependencyManager:
    def __init__(
        self,
        rules,
        lambda_e=0.05,
        lambda_f=0.20,
        hard_on_exist=0.8,
        gray_budget_ratio=0.1,
        strict_hard_constraint=True,
        no_evidence_exist_decay=0.05,
        exist_step=0.10,
        fail_step=0.10,
        fail_credit_mode="independent",
        use_p_exist=True,
        use_p_fail=True,
    ):
        self.rules = rules
        self.lambda_e = float(lambda_e)
        self.lambda_f = float(lambda_f)
        self.hard_on_exist = float(hard_on_exist)
        self.gray_budget_ratio = float(gray_budget_ratio)
        self.strict_hard_constraint = bool(strict_hard_constraint)
        self.no_evidence_exist_decay = max(0.0, float(no_evidence_exist_decay))
        self.exist_step = max(0.0, float(exist_step))
        self.fail_step = max(0.0, float(fail_step))
        self.fail_credit_mode = str(fail_credit_mode).strip().lower()
        self.use_p_exist = self._to_bool(use_p_exist)
        self.use_p_fail = self._to_bool(use_p_fail)

        # Fixed smoothing factor for background failure probability only.
        self.rho = 0.98

        # Background failure probability.
        self.p_bg = 0.1

        # Gray-area accounting for hard-violating samples.
        self.total_hard_decisions = 0
        self.used_gray_decisions = 0
        self.last_update_info = {}

        self.states = {}
        for rule in self.rules:
            rid = rule["id"]
            prior_exist = self._clamp(float(rule.get("prior_exist", 0.5)))
            prior_fail = self._clamp(float(rule.get("prior_fail", rule.get("prior_fail_exist", 0.5))))

            self.states[rid] = {
                "p_exist": prior_exist,
                "p_fail": prior_fail,
                "prior_exist": prior_exist,
                "prior_fail": prior_fail,
                "n_evidence": 0,
                "state": "soft",
            }

    @classmethod
    def from_file(cls, rule_file, **kwargs):
        with open(rule_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            rules = raw.get("rules", [])
        elif isinstance(raw, list):
            rules = raw
        else:
            rules = []

        normalized = []
        for idx, rule in enumerate(rules, 1):
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("id", f"rule_{idx}")).strip()
            if not rid:
                continue
            rtype = str(rule.get("type", "control")).strip().lower()
            if_cond = rule.get("if", {})
            then_cond = rule.get("then", {})
            if not isinstance(if_cond, dict) or not isinstance(then_cond, dict):
                continue
            normalized.append(
                {
                    "id": rid,
                    "type": rtype,
                    "if": if_cond,
                    "then": then_cond,
                    "prior_exist": rule.get("prior_exist", 0.5),
                    "prior_fail": rule.get("prior_fail", rule.get("prior_fail_exist", 0.5)),
                    "probe": rule.get("probe", True),
                    "support_keywords": rule.get("support_keywords", []),
                    "contradict_keywords": rule.get("contradict_keywords", []),
                    "evidence_source": rule.get("evidence_source", ["stdout", "stderr", "system"]),
                    "evidence_lua_path": rule.get("evidence_lua_path", rule.get("lua_path", "")),
                    "evidence_fidelity_overrides": rule.get(
                        "evidence_fidelity_overrides",
                        rule.get("fidelity_overrides", {}),
                    ),
                }
            )
        return cls(normalized, **kwargs)

    def enabled(self):
        return len(self.rules) > 0

    def get_bg_prob(self):
        return self.p_bg

    def get_probs(self, rid):
        st = self.states[rid]
        return st["p_exist"], st["p_fail"]

    def get_effective_probs(self, rid):
        p_exist, p_fail = self.get_probs(rid)
        if not self.use_p_exist:
            p_exist = 1.0
        if not self.use_p_fail:
            p_fail = 1.0
        return p_exist, p_fail

    def violated_rule_ids(self, config):
        violated = []
        for rule in self.rules:
            if self._is_violated(rule, config):
                violated.append(rule["id"])
        return violated

    def penalty(self, config):
        violated = self.violated_rule_ids(config)
        total = 0.0
        risk_by_rule = {}
        for rid in violated:
            p_exist, p_fail = self.get_effective_probs(rid)
            penalty_i = self.lambda_f * p_exist * p_fail
            total += penalty_i
            risk_by_rule[rid] = penalty_i
        return total, violated, risk_by_rule

    def should_reject_hard(self, config):
        violated = self.violated_rule_ids(config)
        hard_violated = [rid for rid in violated if self.states[rid]["state"] == "hard"]
        if not hard_violated:
            return False, []

        if self.strict_hard_constraint:
            return True, hard_violated

        self.total_hard_decisions += 1
        allowed_gray = int(self.gray_budget_ratio * max(1, self.total_hard_decisions))
        if self.used_gray_decisions < allowed_gray:
            self.used_gray_decisions += 1
            return False, hard_violated
        return True, hard_violated

    def update(self, violated_ids, outcome, evidence_by_rule=None):
        evidence_by_rule = evidence_by_rule or {}
        self.last_update_info = {}
        severity = self._severity(outcome)

        # Update background failure only when no dependency is violated.
        if not violated_ids:
            self.p_bg = self._clamp(self.rho * self.p_bg + (1.0 - self.rho) * severity)
            return {}

        credits = self._failure_credits(violated_ids)

        for rid in violated_ids:
            st = self.states[rid]
            evidence_signal = evidence_by_rule.get(rid)
            c_i = credits[rid]
            exist_update = self._update_exist_state(st, evidence_signal)

            fail_update = self._update_fail_state(st, c_i, outcome, severity)

            st["n_evidence"] += 1
            self._update_state(rid)
            self.last_update_info[rid] = {
                "exist_update_reason": exist_update["reason"],
                "p_exist_delta": exist_update["delta"],
                "impact_update_reason": outcome,
                "fail_update_reason": fail_update["reason"],
                "p_fail_delta": fail_update["delta"],
            }

        return credits

    def _failure_credits(self, violated_ids):
        if self.fail_credit_mode == "independent":
            return {rid: 1.0 for rid in violated_ids}

        if self.fail_credit_mode == "uniform":
            uniform = 1.0 / len(violated_ids)
            return {rid: uniform for rid in violated_ids}

        scores = {}
        for rid in violated_ids:
            p_exist, p_fail = self.get_effective_probs(rid)
            scores[rid] = max(1e-6, p_exist * p_fail)

        score_sum = sum(scores.values())
        if score_sum <= 1e-12:
            uniform = 1.0 / len(violated_ids)
            return {rid: uniform for rid in violated_ids}
        return {rid: scores[rid] / score_sum for rid in violated_ids}

    def snapshot_rows(self, iteration, violated_ids, credits, outcome, hard_rejected_ids=None):
        rows = []
        violated_set = set(violated_ids)
        hard_rejected_set = set(hard_rejected_ids or [])
        for rule in self.rules:
            rid = rule["id"]
            p_exist, p_fail = self.get_probs(rid)
            effective_p_exist, effective_p_fail = self.get_effective_probs(rid)
            risk = self.lambda_f * effective_p_exist * effective_p_fail
            st = self.states[rid]
            update_info = self.last_update_info.get(rid, {})
            rows.append(
                {
                    "iter": iteration,
                    "rule_id": rid,
                    "prior_exist": st.get("prior_exist", ""),
                    "prior_fail": st.get("prior_fail", ""),
                    "p_exist": p_exist,
                    "p_fail_exist": p_fail,
                    "p_bg": self.get_bg_prob(),
                    "risk_i": risk,
                    "state": st["state"],
                    "n_evidence": st["n_evidence"],
                    "violated": 1 if rid in violated_set else 0,
                    "hard_rejected": 1 if rid in hard_rejected_set else 0,
                    "credit": float(credits.get(rid, 0.0)),
                    "outcome": outcome,
                    "exist_update_reason": update_info.get("exist_update_reason", "none"),
                    "p_exist_delta": update_info.get("p_exist_delta", 0.0),
                    "impact_update_reason": update_info.get("impact_update_reason", "none"),
                    "fail_update_reason": update_info.get("fail_update_reason", "none"),
                    "p_fail_delta": update_info.get("p_fail_delta", 0.0),
                }
            )
        return rows

    def _update_exist_state(self, st, evidence_signal):
        p_old = st["p_exist"]
        if evidence_signal == "support":
            p_new = self._clamp(p_old + self.exist_step)
            reason = "runtime_support"
        elif evidence_signal == "contradict":
            p_new = self._clamp(p_old - self.exist_step)
            reason = "runtime_contradict"
        elif self.no_evidence_exist_decay > 0:
            p_new = self._clamp(p_old - self.no_evidence_exist_decay)
            reason = "missing_runtime_evidence"
        else:
            p_new = p_old
            reason = "none"
        st["p_exist"] = p_new
        return {"reason": reason, "delta": p_new - p_old}

    def _update_fail_state(self, st, c_i, outcome, severity):
        p_old = st["p_fail"]
        if outcome == "fail":
            delta = self.fail_step * c_i
            reason = "fixed_step_up_fail"
        elif outcome == "degrade":
            delta = 0.5 * self.fail_step * c_i
            reason = "fixed_step_up_degrade"
        else:
            delta = -self.fail_step * c_i
            reason = "fixed_step_down"
        p_new = self._clamp(p_old + delta)
        st["p_fail"] = p_new
        return {"reason": reason, "delta": p_new - p_old}

    def _update_state(self, rid):
        if not self.use_p_exist:
            return
        st = self.states[rid]
        p_exist, _ = self.get_probs(rid)

        if st["state"] == "soft":
            if p_exist + 1e-12 >= self.hard_on_exist:
                st["state"] = "hard"

    def _is_violated(self, rule, config):
        if rule.get("type", "control") != "control":
            return False

        if_cond = rule.get("if", {})
        then_cond = rule.get("then", {})

        # Antecedent must hold.
        for key, expected in if_cond.items():
            if key not in config or not self._matches(config[key], expected, config):
                return False

        # Consequent: all listed fields should satisfy allowed set.
        for key, allowed in then_cond.items():
            allowed_values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            if key not in config:
                return True
            if not any(self._matches(config[key], candidate, config) for candidate in allowed_values):
                return True

        return False

    @staticmethod
    def _matches(actual, expected, config=None):
        if isinstance(expected, dict):
            return DependencyManager._matches_condition(actual, expected, config)

        # Try numeric compare first.
        try:
            return float(actual) == float(expected)
        except (ValueError, TypeError):
            return str(actual) == str(expected)

    @staticmethod
    def _matches_condition(actual, condition, config=None):
        try:
            actual_num = float(actual)
            is_num = True
        except (ValueError, TypeError):
            actual_num = None
            is_num = False

        for op, expected in condition.items():
            op = str(op).strip().lower()
            if op in ("eq", "equals", "=="):
                if not DependencyManager._matches(actual, expected, config):
                    return False
                continue
            if op in ("ne", "!=", "not"):
                if DependencyManager._matches(actual, expected, config):
                    return False
                continue
            if op in ("in", "one_of"):
                values = expected if isinstance(expected, (list, tuple, set)) else [expected]
                if not any(DependencyManager._matches(actual, value, config) for value in values):
                    return False
                continue
            if op in ("not_in", "none_of"):
                values = expected if isinstance(expected, (list, tuple, set)) else [expected]
                if any(DependencyManager._matches(actual, value, config) for value in values):
                    return False
                continue

            if not is_num:
                return False
            expected_num = DependencyManager._resolve_numeric_expected(expected, config)
            if expected_num is None:
                return False
            if op in ("gt", ">") and not actual_num > expected_num:
                return False
            if op in ("gte", "ge", ">=") and not actual_num >= expected_num:
                return False
            if op in ("lt", "<") and not actual_num < expected_num:
                return False
            if op in ("lte", "le", "<=") and not actual_num <= expected_num:
                return False
            if op not in ("gt", ">", "gte", "ge", ">=", "lt", "<", "lte", "le", "<="):
                return False
        return True

    @staticmethod
    def _resolve_numeric_expected(expected, config=None):
        if isinstance(expected, str) and config and expected in config:
            expected = config[expected]
        try:
            return float(expected)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _clamp(v):
        return min(1.0, max(0.0, v))

    @staticmethod
    def _severity(outcome):
        if outcome == "fail":
            return 1.0
        if outcome == "degrade":
            return 0.5
        return 0.0

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
