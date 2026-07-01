import csv
import os
import time
from copy import deepcopy

from tqdm import tqdm

from dependency.dependency_manager import DependencyManager
from dependency.evidence_context import build_context_parts
from dependency.evidence_extractor import RuntimeEvidenceExtractor
from systems.httpddb import HttpdDB
from systems.mysqldb import MysqlDB
from systems.postgresqldb import PostgresqlDB
from systems.tomcatdb import TomcatDB
from systems.x265db import X265DB
from utils.logger import Logger
from utils.path_layout import default_online_rule_file
from workload import WorkloadController


class DependencyAwareTunerBase:
    def __init__(self, args_db, args_workload, args_tune, run, method_name, log_file):
        self.max_iter = int(args_tune["max_iter"])
        self.total_budget = int(args_tune["total_budget"])
        self.args_db = args_db
        self.args_workload = args_workload
        self.args_tune = args_tune
        self.workload_bench = args_workload["workload_bench"]
        self.tuning_method = args_tune["tuning_method"]
        self.fidelity_type = args_tune["fidelity_type"]
        self.fidelity_metric = args_tune["fidelity_metric"]
        self.optimize_objective = args_tune["optimize_objective"]
        self.sys_name = self.args_db["db"]
        self.method_name = method_name
        self.log_file = log_file

        self.dependency_aware = self._to_bool(args_tune.get("dependency_aware", "false"))
        self.run_variant = "depaware" if self.dependency_aware else "plain"
        self.log_path = (
            f"experimental_results/{self.sys_name}/{self.workload_bench}/{self.tuning_method}/"
            f"run_{run}_{self.tuning_method}_{self.fidelity_type}_{self.run_variant}"
        )
        self.cyber_twin_path = f"experimental_results/{self.sys_name}/{self.workload_bench}"
        self.cyber_twin_file = "cyber-twin.csv"

        self.consumed_cost = 0
        self.consumed_iters = 0
        self.evaluated_configs = set()
        self.consecutive_failures = 0
        self.fail_fast = self._to_bool(args_tune.get("fail_fast", "true"))
        self.fail_streak = int(args_tune.get("fail_streak", 3))
        self.fail_action = str(args_tune.get("fail_action", "recover")).strip().lower()
        self.dep_iter = 0
        self.dep_degrade_ratio = float(args_tune.get("dep_degrade_ratio", 0.10))
        self.dep_ema_alpha = float(args_tune.get("dep_ema_alpha", 0.30))
        self.dep_probe_budget_ratio = float(args_tune.get("dep_probe_budget_ratio", 0.0))
        self.dep_probe_effect_threshold = float(args_tune.get("dep_probe_effect_threshold", self.dep_degrade_ratio))
        self.dep_probe_variants_per_rule = int(args_tune.get("dep_probe_variants_per_rule", 3))
        self.dep_candidate_batch_size = int(args_tune.get("dep_candidate_batch_size", 16))
        self.perf_baseline = None

        self.target_system = self._init_target_system()
        self.workload_controller = WorkloadController(args_db, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)
        self.dep_manager = self._init_dependency_manager()
        self.dependency_aware = self.dep_manager is not None
        self.evidence_extractor = None
        if self.dependency_aware:
            self.evidence_extractor = RuntimeEvidenceExtractor(
                self.workload_controller,
                self.target_system,
                self.dep_manager,
            )
        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="iter")

    def _init_target_system(self):
        if self.sys_name == "mysql":
            return MysqlDB(self.args_db)
        if self.sys_name == "postgresql":
            return PostgresqlDB(self.args_db)
        if self.sys_name == "httpd":
            return HttpdDB(self.args_db)
        if self.sys_name == "tomcat":
            return TomcatDB(self.args_db)
        if self.sys_name == "x265":
            return X265DB(self.args_db)
        raise ValueError(f"Unsupported system: {self.sys_name}")

    def _init_dependency_manager(self):
        if not self.dependency_aware:
            return None
        rule_file = str(self.args_tune.get("dependency_rule_file", "")).strip()
        if not rule_file:
            rule_file = default_online_rule_file(self.sys_name)
        if not os.path.exists(rule_file):
            print(f"[DependencyAware] Rule file not found: {rule_file}; dependency aware disabled.")
            return None
        manager = DependencyManager.from_file(
            rule_file,
            lambda_e=float(self.args_tune.get("dep_lambda_e", 0.05)),
            lambda_f=float(self.args_tune.get("dep_lambda_f", 0.20)),
            hard_on_exist=float(self.args_tune.get("dep_hard_on_exist", 0.8)),
            gray_budget_ratio=float(self.args_tune.get("dep_gray_budget_ratio", 0.0)),
            strict_hard_constraint=self._to_bool(self.args_tune.get("dep_strict_hard_constraint", "true")),
            no_evidence_exist_decay=float(self.args_tune.get("dep_no_evidence_exist_decay", 0.05)),
            exist_step=float(self.args_tune.get("dep_exist_step", 0.10)),
            fail_step=float(self.args_tune.get("dep_fail_step", 0.10)),
            fail_credit_mode=self.args_tune.get("dep_fail_credit_mode", "independent"),
            use_p_exist=self._to_bool(self.args_tune.get("dep_use_p_exist", "true")),
            use_p_fail=self._to_bool(self.args_tune.get("dep_use_p_fail", "true")),
        )
        return manager if manager.enabled() else None

    def tune_single_fidelity(self, search_fn):
        start_time = time.time()
        try:
            self._preflight_db_or_raise()
            if self.fidelity_type == "single_fidelity":
                hf_factors = self.workload_controller.get_default_fidelity_factors()
                probe_iters = self.dep_probe_warmup(hf_factors, observe_fn=getattr(self, "observe_probe_config", None))
                search_fn(self.total_budget, hf_factors, start_consumed=probe_iters)
            elif self.fidelity_type == "multi_fidelity":
                raise NotImplementedError(f"{self.method_name} currently supports single_fidelity only.")
        finally:
            self._restore_system_default_state()
            runtime = time.time() - start_time
            self.logger.store_runtime_to_csv(runtime, self.log_path)
            self.pbar.close()

    def evaluate_configs(self, configs, factors, stage_budget=None):
        evaluated_configs = []
        cost_configs = 0
        current_loop_consumption = 0
        for config in configs:
            knob_apply_ok = self.target_system.set_db_knob(config)
            num_iter = 1
            total_perf = total_prepare_time = total_run_time = total_clean_time = total_evaluated_cost = 0
            total_latency = total_throughput = total_qps = 0
            for _ in range(num_iter):
                start = time.time()
                try:
                    if knob_apply_ok is False:
                        raise RuntimeError("knob apply failed")
                    print(f"Execute workload: {factors}")
                    latency, throughput, qps, prepare_time, run_time, clean_time = self.workload_controller.run_workload(factors)
                except Exception as e:
                    print(f"Workload execution failed: {e}")
                    latency = throughput = qps = prepare_time = run_time = clean_time = 0
                evaluated_cost = time.time() - start
                if self.optimize_objective == "throughput":
                    total_perf += throughput
                elif self.optimize_objective == "latency":
                    total_perf += latency
                elif self.optimize_objective == "qps":
                    total_perf += qps
                elif self.optimize_objective in ["EncodeFPS", "encode_fps"]:
                    total_perf += qps
                total_latency += latency
                total_throughput += throughput
                total_qps += qps
                total_prepare_time += prepare_time
                total_run_time += run_time
                total_clean_time += clean_time
                total_evaluated_cost += evaluated_cost

            perf = total_perf / num_iter
            prepare_time = total_prepare_time / num_iter
            run_time = total_run_time / num_iter
            clean_time = total_clean_time / num_iter
            evaluated_cost = total_evaluated_cost / num_iter
            avg_latency = total_latency / num_iter
            avg_throughput = total_throughput / num_iter
            avg_qps = total_qps / num_iter

            if self._is_failed_eval(perf, avg_qps, avg_throughput, avg_latency):
                self.consecutive_failures += 1
                print(f"[FailFast] invalid evaluation detected. consecutive_failures={self.consecutive_failures}")
            else:
                self.consecutive_failures = 0

            self.pbar.update(1)
            print()
            print(f"[PERFORMANCE]: {self.optimize_objective}: {perf}")
            print("-------------------------------------------------")

            self.consumed_cost += evaluated_cost
            self.consumed_iters += 1
            current_loop_consumption += evaluated_cost
            cost_configs += evaluated_cost
            evaluated_configs.append((config, perf, evaluated_cost))

            self.logger.logging_data(
                config,
                perf,
                evaluated_cost,
                factors,
                prepare_time,
                run_time,
                clean_time,
                self.log_path,
                self.log_file,
                latency=avg_latency,
                throughput=avg_throughput,
                qps=avg_qps,
            )
            self.logger.logging_cyber_twin(
                config,
                perf,
                evaluated_cost,
                factors,
                prepare_time,
                run_time,
                clean_time,
                self.cyber_twin_path,
                self.cyber_twin_file,
            )

            if self.fail_fast and self.consecutive_failures >= self.fail_streak:
                self._handle_failure_streak(factors)
            if self.consumed_iters >= self.total_budget:
                break
            if stage_budget is not None and current_loop_consumption >= stage_budget:
                print("stage budge exhausted.")
                break
        return evaluated_configs, cost_configs

    def dependency_adjusted_perf(self, config, raw_perf, violated_ids, penalty, factors=None):
        perf = raw_perf
        if not self.dependency_aware:
            return perf
        perf = self.apply_dependency_penalty(raw_perf, penalty)

        outcome = self.classify_outcome(raw_perf)
        evidence_by_rule = self.extract_runtime_evidence(violated_ids, config=config, factors=factors)
        credits = self.dep_manager.update(violated_ids, outcome, evidence_by_rule=evidence_by_rule)
        self.dep_iter += 1
        self.log_dependency_snapshot(violated_ids, credits, outcome, evidence_by_rule)
        return perf

    def apply_dependency_penalty(self, raw_perf, penalty):
        if self.optimize_objective in ["throughput", "RPS", "qps", "EncodeFPS", "encode_fps"]:
            return raw_perf - penalty
        if self.optimize_objective == "latency":
            return raw_perf + penalty
        return raw_perf

    def objective_value(self, perf):
        if self.optimize_objective in ["throughput", "RPS", "qps", "EncodeFPS", "encode_fps"]:
            return -perf
        return perf

    def config_key(self, config, fidelity):
        return tuple(sorted(config.items())), tuple(sorted(fidelity.items()))

    def _is_failed_eval(self, perf, qps, throughput, latency):
        if self.optimize_objective in ["qps", "throughput", "RPS", "EncodeFPS", "encode_fps"]:
            return perf <= 0 or qps <= 0 or throughput <= 0
        if self.optimize_objective == "latency":
            return latency <= 0
        return perf <= 0

    def _preflight_db_or_raise(self):
        max_retry = 5
        for i in range(1, max_retry + 1):
            try:
                if self.target_system.check_connection_alive():
                    print(f"[Preflight] DB connectivity check passed at attempt {i}/{max_retry}.")
                    return
            except Exception as e:
                print(f"[Preflight] DB connectivity check exception at attempt {i}/{max_retry}: {e}")
            if i == 1:
                self._restore_system_default_state(silent=True)
            time.sleep(1)
        raise RuntimeError(f"Preflight failed: cannot connect to target system '{self.sys_name}' before tuning starts.")

    def _restore_system_default_state(self, silent=False):
        restore = getattr(self.target_system, "restore_config", None)
        restart = getattr(self.target_system, "restart_container", None)
        if restore is None:
            return
        try:
            restore()
            if restart is not None:
                restart()
        except Exception as e:
            if not silent:
                print(f"[Cleanup] failed to restore default state for {self.sys_name}: {e}")

    def _handle_failure_streak(self, factors):
        message = (
            f"{self.consecutive_failures} consecutive invalid evaluations "
            f"(objective={self.optimize_objective})."
        )
        if self.fail_action in ("abort", "raise", "stop"):
            raise RuntimeError(f"Fail-fast triggered: {message}")

        print(f"[FailRecovery] {message} Recovering target system and continuing search.")
        self._restore_system_default_state(silent=False)
        safe_config = self._failure_recovery_config()
        if safe_config:
            try:
                ok = self.target_system.set_db_knob(safe_config)
                print(f"[FailRecovery] Applied conservative recovery config: ok={ok}")
            except Exception as e:
                print(f"[FailRecovery] Failed to apply conservative recovery config: {e}")
                self._restore_system_default_state(silent=True)
        self.consecutive_failures = 0

    def _failure_recovery_config(self):
        mode = str(self.args_tune.get("fail_recovery_config", "default")).strip().lower()
        if mode in ("none", "off", "false"):
            return None
        config = self.target_system.get_default_knobs()
        raw = str(self.args_tune.get("fail_safe_defaults", "")).strip()
        if raw:
            for item in raw.split(","):
                if "=" not in item:
                    continue
                knob, raw_value = item.split("=", 1)
                knob = knob.strip()
                if not knob:
                    continue
                config[knob] = self._cast_knob_value(knob, raw_value.strip())
        return config

    def _cast_knob_value(self, knob, value):
        info = self.target_system.knobs_info.get(knob, {})
        ktype = str(info.get("type", "")).lower()
        if ktype in ("integer", "int"):
            return int(float(value))
        if ktype in ("float", "num"):
            return float(value)
        return value

    def classify_outcome(self, perf):
        if perf <= 0:
            return "fail"
        if self.perf_baseline is None:
            self.perf_baseline = perf
            return "ok"
        outcome = "ok"
        if self.optimize_objective in ["throughput", "qps", "RPS", "EncodeFPS", "encode_fps"]:
            if perf < (1.0 - self.dep_degrade_ratio) * self.perf_baseline:
                outcome = "degrade"
        elif self.optimize_objective == "latency":
            if perf > (1.0 + self.dep_degrade_ratio) * self.perf_baseline:
                outcome = "degrade"
        self.perf_baseline = (
            (1.0 - self.dep_ema_alpha) * self.perf_baseline + self.dep_ema_alpha * perf
        )
        return outcome

    def extract_runtime_evidence(self, violated_ids, config=None, factors=None):
        if not violated_ids or not self.dep_manager:
            return {}
        if self.evidence_extractor is not None:
            extra_parts = build_context_parts(
                self.sys_name,
                violated_ids,
                config,
                factors,
                workload_controller=self.workload_controller,
            )
            detailed = self.evidence_extractor.extract_with_details(
                violated_ids,
                extra_parts=extra_parts,
                include_live_sources=True,
            )
            return {
                rid: detail["signal"]
                for rid, detail in detailed["details"].items()
                if detail["signal"] != "none"
            }

        raw_parts = []
        if self.workload_controller.output is not None:
            try:
                raw_parts.append(self.workload_controller.output.decode("utf-8", errors="ignore"))
            except Exception:
                raw_parts.append(str(self.workload_controller.output))
        if self.workload_controller.error is not None:
            try:
                raw_parts.append(self.workload_controller.error.decode("utf-8", errors="ignore"))
            except Exception:
                raw_parts.append(str(self.workload_controller.error))
        raw_text = "\n".join(raw_parts).lower()

        evidence = {}
        for rule in self.dep_manager.rules:
            rid = rule["id"]
            if rid not in violated_ids:
                continue
            support_keywords = [str(x).lower() for x in rule.get("support_keywords", [])]
            contradict_keywords = [str(x).lower() for x in rule.get("contradict_keywords", [])]
            has_support = self._contains_any(raw_text, support_keywords)
            has_contradict = self._contains_any(raw_text, contradict_keywords)
            if has_support and not has_contradict:
                evidence[rid] = "support"
            elif has_contradict and not has_support:
                evidence[rid] = "contradict"
        return evidence

    def dep_probe_warmup(self, fidelity, start_consumed=0, observe_fn=None):
        if not self.dependency_aware or not self.dep_manager:
            return 0

        remaining = max(0, self.total_budget - int(start_consumed))
        probe_budget = min(remaining, int(self.total_budget * self.dep_probe_budget_ratio))
        if probe_budget < 2:
            return 0

        base_config = self.target_system.get_default_knobs()
        probe_configs = [("dep_probe_control", None, deepcopy(base_config))]
        for rule in self.dep_manager.rules:
            if len(probe_configs) >= probe_budget:
                break
            if not self._to_bool(rule.get("probe", True)):
                continue
            configs = self.build_rule_violation_configs(
                base_config,
                rule,
                max_variants=max(1, self.dep_probe_variants_per_rule),
            )
            for variant_idx, config in enumerate(configs, 1):
                if len(probe_configs) >= probe_budget:
                    break
                if self.config_key(config, fidelity) in self.evaluated_configs:
                    continue
                probe_configs.append((f"dep_probe_violate::{rule['id']}::v{variant_idx}", rule["id"], config))

        if len(probe_configs) <= 1:
            return 0

        print(
            f"[DependencyProbe] start warmup: budget={len(probe_configs)}/{self.total_budget}, "
            f"ratio={self.dep_probe_budget_ratio}"
        )

        consumed = 0
        control_perf = None
        for label, target_rule_id, config in probe_configs:
            evaluated_config, _ = self.evaluate_configs([config], fidelity)
            if not evaluated_config:
                continue

            observed_config, raw_perf, _ = evaluated_config[0]
            violated_ids = self.dep_manager.violated_rule_ids(observed_config)
            penalty, _, _ = self.dep_manager.penalty(observed_config)
            adjusted_perf = self.apply_dependency_penalty(raw_perf, penalty)
            if observe_fn is not None:
                observe_fn(observed_config, adjusted_perf)
            self.evaluated_configs.add(self.config_key(observed_config, fidelity))
            consumed += 1

            if target_rule_id is None:
                control_perf = raw_perf
                if raw_perf > 0:
                    self.perf_baseline = raw_perf
                credits = self.dep_manager.update([], "ok")
                self.dep_iter += 1
                self.log_dependency_snapshot([], credits, "ok", {})
                self.log_dep_probe_row(label, "", raw_perf, "", "ok", "none", violated_ids)
                continue

            outcome, relative_change = self.classify_probe_result(control_perf, raw_perf)
            evidence_by_rule = self.extract_runtime_evidence([target_rule_id], config=observed_config, factors=fidelity)
            evidence_signal = evidence_by_rule.get(target_rule_id, "none")
            credits = self.dep_manager.update(
                [target_rule_id],
                outcome,
                evidence_by_rule=evidence_by_rule,
            )
            self.dep_iter += 1
            self.log_dependency_snapshot([target_rule_id], credits, outcome, evidence_by_rule)
            self.log_dep_probe_row(
                label,
                target_rule_id,
                raw_perf,
                relative_change,
                outcome,
                evidence_signal,
                violated_ids,
            )

        print(f"[DependencyProbe] completed: consumed={consumed}")
        return consumed

    def classify_probe_result(self, control_perf, perf):
        if perf <= 0:
            return "fail", ""
        if control_perf is None or abs(control_perf) <= 1e-12:
            return "ok", ""

        relative_change = (perf - control_perf) / abs(control_perf)
        threshold = self.dep_probe_effect_threshold
        if self.optimize_objective in ["throughput", "RPS", "qps", "EncodeFPS", "encode_fps"]:
            if relative_change <= -threshold:
                return "degrade", relative_change
        elif self.optimize_objective == "latency":
            if relative_change >= threshold:
                return "degrade", relative_change
        return "ok", relative_change

    def log_dependency_snapshot(self, violated_ids, credits, outcome, evidence_by_rule, hard_rejected_ids=None):
        dep_rows = self.dep_manager.snapshot_rows(
            self.dep_iter,
            violated_ids,
            credits,
            outcome,
            hard_rejected_ids=hard_rejected_ids,
        )
        for row in dep_rows:
            row["evidence_signal"] = evidence_by_rule.get(row["rule_id"], "none")
            self.logger.logging_dependency_state(row, self.log_path, "dep_state.csv")

    def log_dep_probe_row(self, label, rule_id, perf, relative_change, outcome, evidence_signal, violated_ids):
        os.makedirs(self.log_path, exist_ok=True)
        file_path = os.path.join(self.log_path, "dep_probe.csv")
        file_exists = os.path.exists(file_path)
        header = [
            "label",
            "rule_id",
            "perf",
            "relative_change",
            "outcome",
            "existence_evidence_signal",
            "violated_ids",
        ]
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "label": label,
                    "rule_id": rule_id,
                    "perf": perf,
                    "relative_change": relative_change,
                    "outcome": outcome,
                    "existence_evidence_signal": evidence_signal,
                    "violated_ids": "|".join(violated_ids),
                }
            )

    def build_rule_violation_configs(self, base_config, rule, max_variants=3):
        if rule.get("type", "control") != "control":
            return []

        anchor_config = deepcopy(base_config)
        for knob, expected in rule.get("if", {}).items():
            value = self.pick_condition_value(knob, expected)
            if value is None:
                return []
            anchor_config[knob] = value

        violating_by_knob = {}
        max_candidates = 0
        for knob, allowed in rule.get("then", {}).items():
            candidates = self.pick_violating_values(knob, allowed, anchor_config, max_variants=max_variants)
            if not candidates:
                return []
            violating_by_knob[knob] = candidates
            max_candidates = max(max_candidates, len(candidates))

        configs = []
        seen = set()
        for idx in range(min(max_variants, max_candidates)):
            config = deepcopy(anchor_config)
            changed = False
            for knob, candidates in violating_by_knob.items():
                config[knob] = candidates[min(idx, len(candidates) - 1)]
                changed = True
            if not changed or rule["id"] not in self.dep_manager.violated_rule_ids(config):
                continue
            key = tuple(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(config)
        return configs

    def pick_condition_value(self, knob, expected):
        for value in self.knob_candidates(knob):
            if DependencyManager._matches(value, expected):
                return value

        if isinstance(expected, dict):
            info = self.target_system.knobs_info.get(knob, {})
            min_value = info.get("min")
            max_value = info.get("max")
            for op, target in expected.items():
                op = str(op).strip().lower()
                if op in ("eq", "equals", "=="):
                    return target
                if op in ("in", "one_of"):
                    values = target if isinstance(target, (list, tuple, set)) else [target]
                    return next(iter(values), None)
                if op in ("lt", "<", "lte", "le", "<=") and min_value is not None:
                    return min_value
                if op in ("gt", ">", "gte", "ge", ">=") and max_value is not None:
                    return max_value
            return None
        return expected

    def pick_violating_values(self, knob, allowed, config, max_variants=3):
        allowed_values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        violating = []
        for value in self.knob_candidates(knob):
            if not any(DependencyManager._matches(value, candidate, config) for candidate in allowed_values):
                violating.append(value)
        if not violating:
            return []
        violating.sort(
            key=lambda value: self.violation_strength(knob, value, allowed_values, config)
        )
        return violating[:max(1, max_variants)]

    def violation_strength(self, knob, value, allowed_values, config):
        info = self.target_system.knobs_info.get(knob, {})
        min_value = info.get("min")
        max_value = info.get("max")
        try:
            value_num = float(value)
            is_num = True
        except (TypeError, ValueError):
            value_num = None
            is_num = False

        strengths = []
        for allowed in allowed_values:
            if not isinstance(allowed, dict):
                if is_num:
                    try:
                        strengths.append(abs(value_num - float(allowed)))
                    except (TypeError, ValueError):
                        continue
                continue
            for op, expected in allowed.items():
                expected_num = DependencyManager._resolve_numeric_expected(expected, config)
                if expected_num is None or not is_num:
                    continue
                op = str(op).strip().lower()
                if op in ("gt", ">"):
                    strengths.append(max(0.0, expected_num - value_num))
                elif op in ("gte", "ge", ">="):
                    strengths.append(max(0.0, expected_num - value_num + 1e-9))
                elif op in ("lt", "<"):
                    strengths.append(max(0.0, value_num - expected_num))
                elif op in ("lte", "le", "<="):
                    strengths.append(max(0.0, value_num - expected_num + 1e-9))
                elif op in ("eq", "equals", "=="):
                    strengths.append(abs(value_num - expected_num))
                elif op in ("in", "one_of"):
                    values = expected if isinstance(expected, (list, tuple, set)) else [expected]
                    numeric_values = []
                    for item in values:
                        resolved = DependencyManager._resolve_numeric_expected(item, config)
                        if resolved is not None:
                            numeric_values.append(abs(value_num - resolved))
                    if numeric_values:
                        strengths.append(min(numeric_values))

        if strengths:
            strength = min(strengths)
            if min_value is not None and max_value is not None:
                try:
                    knob_range = max(1e-9, float(max_value) - float(min_value))
                    return (0, strength / knob_range)
                except (TypeError, ValueError):
                    return (0, strength)
            return (0, strength)
        return (1, str(value))

    def knob_candidates(self, knob):
        info = self.target_system.knobs_info.get(knob, {})
        candidates = []
        for key in ("default", "min", "max"):
            if info.get(key) is not None:
                candidates.append(info.get(key))
        candidates.extend(info.get("enum_values", []) or [])

        if str(info.get("type", "")).lower() in ("integer", "int"):
            try:
                min_value = int(info.get("min"))
                max_value = int(info.get("max"))
                default_value = int(info.get("default", min_value))
                midpoint = int((min_value + max_value) / 2)
                lower_mid = int((min_value + default_value) / 2)
                upper_mid = int((default_value + max_value) / 2)
                candidates.extend(
                    [min_value, min_value + 1, lower_mid, default_value, midpoint, upper_mid, max_value - 1, max_value]
                )
            except (TypeError, ValueError):
                pass
        elif str(info.get("type", "")).lower() in ("float", "num"):
            try:
                min_value = float(info.get("min"))
                max_value = float(info.get("max"))
                default_value = float(info.get("default", min_value))
                span = max_value - min_value
                candidates.extend(
                    [
                        min_value,
                        min_value + 0.1 * span,
                        min_value + 0.25 * span,
                        default_value,
                        min_value + 0.5 * span,
                        min_value + 0.75 * span,
                        min_value + 0.9 * span,
                        max_value,
                    ]
                )
            except (TypeError, ValueError):
                pass

        result = []
        seen = set()
        for value in candidates:
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _contains_any(text, keywords):
        for kw in keywords:
            if kw and kw in text:
                return True
        return False
