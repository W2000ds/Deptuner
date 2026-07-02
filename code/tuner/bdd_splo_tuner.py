import csv
import math
import os
import random

from tuner.depaware_tuner_base import DependencyAwareTunerBase
from utils.bdd_feature_space import BDDFeatureSpace
from utils.knob_space_utils import sanitize_config_for_optimizer
from utils.path_layout import default_online_rule_file


class BDDSPLTuner(DependencyAwareTunerBase):
    def __init__(self, args_db, args_workload, args_tune, run):
        method = str(args_tune["tuning_method"]).strip().lower()
        if method not in ("bdd_srs", "bdd_rrs"):
            raise ValueError(f"Unsupported BDD SPLO method: {method}")
        super().__init__(args_db, args_workload, args_tune, run, method, "BDDSplotuner_results.csv")
        self.method = method
        self.rng = random.Random(int(run))
        self.history = []
        self.search_evals = 0
        self.best_perf = None
        self.best_objective = None
        self.best_config = None
        self.rrs_round_size = int(args_tune.get("rrs_round_size", 30))
        self.bdd_numeric_bins = int(args_tune.get("bdd_numeric_bins", 5))
        self.bdd_common_top_k = int(args_tune.get("bdd_common_top_k", 2))
        self.bdd_stop_on_convergence = self._to_bool(args_tune.get("bdd_stop_on_convergence", "false"))
        self.bdd_use_dependency_rules = self._to_bool(args_tune.get("bdd_use_dependency_rules", "true"))
        self.bdd_eval_default_first = self._to_bool(args_tune.get("bdd_eval_default_first", "true"))
        rule_file = self.args_tune.get("dependency_rule_file", "") if self.bdd_use_dependency_rules else ""
        if self.bdd_use_dependency_rules and not str(rule_file).strip():
            rule_file = default_online_rule_file(self.sys_name)
        self.feature_space = BDDFeatureSpace(
            self.target_system.knobs_info,
            rule_file=rule_file,
            numeric_bins=self.bdd_numeric_bins,
            seed=int(run),
        )
        self._log_feature_space_summary()

    def tune_bdd_srs(self):
        self.tune_single_fidelity(self.bdd_srs_search_config)

    def tune_bdd_rrs(self):
        self.tune_single_fidelity(self.bdd_rrs_search_config)

    def bdd_srs_search_config(self, budget, fidelity, start_consumed=0):
        consumed_iters = int(start_consumed)
        consumed_iters += self._evaluate_default_first(fidelity, budget, consumed_iters)
        while consumed_iters < budget:
            config = self._next_bdd_config(fidelity, fixed={})
            if config is None:
                print("[BDD-SRS] no legal unevaluated candidate found; stop search.")
                break
            evaluated = self._evaluate_candidate(config, fidelity)
            if evaluated is None:
                continue
            consumed_iters += evaluated

    def bdd_rrs_search_config(self, budget, fidelity, start_consumed=0):
        consumed_iters = int(start_consumed)
        fixed = {}
        consumed_iters += self._evaluate_default_first(fidelity, budget, consumed_iters)

        while consumed_iters < budget:
            remaining = budget - consumed_iters
            round_size = max(1, min(self.rrs_round_size, remaining))
            round_records = []

            for _ in range(round_size):
                config = self._next_bdd_config(fidelity, fixed=fixed)
                if config is None:
                    print("[BDD-RRS] no legal unevaluated candidate found; stop current round.")
                    break
                evaluated = self._evaluate_candidate(config, fidelity)
                if evaluated is None:
                    continue
                consumed_iters += evaluated
                round_records.append(self.history[-1])
                if consumed_iters >= budget:
                    break

            if len(round_records) < 2:
                break

            next_fixed = dict(fixed)
            top_k = max(2, min(self.bdd_common_top_k, len(round_records)))
            ordered = sorted(round_records, key=lambda row: row["objective"])
            common = self.feature_space.common_features([row["config"] for row in ordered[:top_k]])
            for knob, value in common.items():
                next_fixed.setdefault(knob, value)

            changed = next_fixed != fixed
            self._log_rrs_round(len(round_records), fixed, next_fixed, changed)
            if not changed and self.bdd_stop_on_convergence:
                break
            fixed = next_fixed

    def _evaluate_default_first(self, fidelity, budget, consumed_iters):
        if not self.bdd_eval_default_first or consumed_iters >= budget:
            return 0
        default_config = self.target_system.get_default_knobs()
        fixed_default = {
            name: value
            for name, value in default_config.items()
            if name in self.feature_space.domains
        }
        config = self.feature_space.sample(fixed=fixed_default)
        if config is None:
            config = {
                name: default_config[name]
                for name in self.feature_space.names
                if name in default_config
            }
        if not config:
            return 0
        evaluated = self._evaluate_candidate(config, fidelity)
        return int(evaluated or 0)

    def _next_bdd_config(self, fidelity, fixed):
        attempts = int(self.args_tune.get("bdd_sample_max_attempts", 1000))
        for _ in range(attempts):
            config = self.feature_space.sample(fixed=fixed)
            if config is None:
                return None
            if self.config_key(config, fidelity) in self.evaluated_configs:
                continue
            return config
        return None

    def _evaluate_candidate(self, config, fidelity):
        violated_ids = []
        penalty = 0.0
        if self.dependency_aware:
            reject, _ = self.dep_manager.should_reject_hard(config)
            if reject:
                return None
            penalty, violated_ids, _ = self.dep_manager.penalty(config)

        key = self.config_key(config, fidelity)
        if key in self.evaluated_configs:
            return None

        evaluated_config, _ = self.evaluate_configs([config], fidelity)
        if not evaluated_config:
            return None

        observed_config, raw_perf, _ = evaluated_config[0]
        adjusted_perf = self.dependency_adjusted_perf(
            observed_config,
            raw_perf,
            violated_ids,
            penalty,
            fidelity,
        )
        objective = self.objective_value(adjusted_perf)
        normalized = sanitize_config_for_optimizer(self.target_system.knobs_info, observed_config)
        self.history.append(
            {
                "config": normalized,
                "raw_perf": raw_perf,
                "adjusted_perf": adjusted_perf,
                "objective": objective,
            }
        )
        self.evaluated_configs.add(key)
        self.search_evals += 1
        self._update_best(normalized, adjusted_perf, objective)
        self._log_sampling_stats()
        return len(evaluated_config)

    def _update_best(self, config, perf, objective):
        if self.best_objective is None or objective < self.best_objective:
            self.best_config = dict(config)
            self.best_perf = perf
            self.best_objective = objective

    def _log_sampling_stats(self):
        n = self.search_evals
        if n <= 0:
            return
        row = {
            "iter": n,
            "method": self.method,
            "best_perf": self.best_perf,
            "legal_configurations": self.feature_space.count(),
            "expected_rank": 1.0 / (n + 1),
            "expected_rank_percent": 100.0 / (n + 1),
            "std_rank": math.sqrt(2.0 / ((n + 1) * (n + 2)) - (1.0 / (n + 1)) ** 2),
            "confidence_top_1_percent": 1.0 - math.pow(1.0 - 0.01, n),
            "confidence_top_2_percent": 1.0 - math.pow(1.0 - 0.02, n),
        }
        self._append_csv("bdd_sampling_stats.csv", row)

    def _log_feature_space_summary(self):
        summary = self.feature_space.feature_summary()
        summary.update(
            {
                "method": self.method,
                "numeric_bins": self.bdd_numeric_bins,
                "dependency_rules_enabled": self.bdd_use_dependency_rules,
            }
        )
        self._append_csv("bdd_feature_space.csv", summary)

    def _log_rrs_round(self, round_size, fixed, next_fixed, changed):
        row = {
            "search_evals": self.search_evals,
            "round_size": round_size,
            "changed": changed,
            "fixed": self._fixed_summary(fixed),
            "next_fixed": self._fixed_summary(next_fixed),
            "best_perf": self.best_perf,
            "remaining_legal_configurations": self.feature_space.count(next_fixed),
        }
        self._append_csv("bdd_rrs_rounds.csv", row)

    def _append_csv(self, filename, row):
        os.makedirs(self.log_path, exist_ok=True)
        path = os.path.join(self.log_path, filename)
        file_exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _fixed_summary(fixed):
        return ";".join(f"{k}={v}" for k, v in sorted(fixed.items()))
