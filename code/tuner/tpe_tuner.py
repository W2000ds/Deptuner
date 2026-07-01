import random

import numpy as np
from sklearn.neighbors import KernelDensity

from tuner.depaware_tuner_base import DependencyAwareTunerBase
from utils.knob_space_utils import build_categorical_values, canonicalize_value, sanitize_config_for_optimizer


class TPETuner(DependencyAwareTunerBase):
    def __init__(self, args_db, args_workload, args_tune, run):
        super().__init__(args_db, args_workload, args_tune, run, "tpe", "TPETuner_results.csv")
        self.rng = random.Random(int(run))
        self.n_startup = int(args_tune.get("tpe_n_startup", 20))
        self.gamma = float(args_tune.get("tpe_gamma", 0.20))
        self.n_candidates = int(args_tune.get("tpe_n_candidates", 256))
        self.history = []
        self.space = self._build_space()

    def tune_tpe(self):
        self.tune_single_fidelity(self.tpe_search_config)

    def tpe_search_config(self, budget, fidelity, start_consumed=0):
        consumed_iters = int(start_consumed)
        while consumed_iters < budget:
            config = self._suggest()

            violated_ids = []
            penalty = 0.0
            if self.dependency_aware:
                reject, _ = self.dep_manager.should_reject_hard(config)
                if reject:
                    continue
                penalty, violated_ids, _ = self.dep_manager.penalty(config)

            if self.config_key(config, fidelity) in self.evaluated_configs:
                continue

            evaluated_config, _ = self.evaluate_configs([config], fidelity)
            if not evaluated_config:
                continue
            raw_perf = evaluated_config[0][1]
            perf = self.dependency_adjusted_perf(config, raw_perf, violated_ids, penalty, fidelity)
            objective = self.objective_value(perf)
            normalized = sanitize_config_for_optimizer(self.target_system.knobs_info, config)
            self.history.append((normalized, objective))
            self.evaluated_configs.add(self.config_key(config, fidelity))
            consumed_iters += len(evaluated_config)

    def observe_probe_config(self, config, adjusted_perf):
        normalized = sanitize_config_for_optimizer(self.target_system.knobs_info, config)
        self.history.append((normalized, self.objective_value(adjusted_perf)))

    def _build_space(self):
        space = []
        for name, info in self.target_system.knobs_info.items():
            ktype = str(info.get("type", "")).lower()
            if ktype == "integer":
                space.append({"name": name, "type": "integer", "min": int(info["min"]), "max": int(info["max"])})
            elif ktype == "float":
                space.append({"name": name, "type": "float", "min": float(info["min"]), "max": float(info["max"])})
            elif ktype in ("enum", "boolean"):
                categories = build_categorical_values(info)
                space.append({"name": name, "type": "categorical", "categories": categories})
        return space

    def _suggest(self):
        if len(self.history) < self.n_startup:
            if self.dependency_aware:
                candidates = [self._sample_random() for _ in range(max(1, self.dep_candidate_batch_size))]
                scored = []
                for candidate in candidates:
                    reject, _ = self.dep_manager.should_reject_hard(candidate)
                    if reject:
                        continue
                    penalty, _, _ = self.dep_manager.penalty(candidate)
                    scored.append((-penalty, candidate))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return scored[0][1]
            return self._sample_random()

        candidates = [self._sample_random() for _ in range(self.n_candidates)]
        scores = []
        for candidate in candidates:
            score = self._density_ratio_score(candidate)
            if self.dependency_aware:
                reject, _ = self.dep_manager.should_reject_hard(candidate)
                if reject:
                    continue
                penalty, _, _ = self.dep_manager.penalty(candidate)
                score -= penalty
            scores.append((score, candidate))
        if not scores:
            return self._sample_random()
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def _sample_random(self):
        config = {}
        for dim in self.space:
            if dim["type"] == "integer":
                config[dim["name"]] = self.rng.randint(dim["min"], dim["max"])
            elif dim["type"] == "float":
                config[dim["name"]] = self.rng.uniform(dim["min"], dim["max"])
            else:
                config[dim["name"]] = self.rng.choice(dim["categories"])
        return config

    def _density_ratio_score(self, config):
        encoded_history = np.array([self._encode(c) for c, _ in self.history], dtype=float)
        objectives = np.array([obj for _, obj in self.history], dtype=float)
        n_good = max(1, int(np.ceil(self.gamma * len(self.history))))
        order = np.argsort(objectives)
        good = encoded_history[order[:n_good]]
        bad = encoded_history[order[n_good:]]
        if len(bad) < 2:
            return self.rng.random()

        x = np.array([self._encode(config)], dtype=float)
        bandwidth = max(0.05, 1.0 / np.sqrt(len(self.history)))
        good_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(good)
        bad_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(bad)
        return float(good_kde.score_samples(x)[0] - bad_kde.score_samples(x)[0])

    def _encode(self, config):
        normalized = sanitize_config_for_optimizer(self.target_system.knobs_info, config)
        values = []
        for dim in self.space:
            info = self.target_system.knobs_info[dim["name"]]
            value = canonicalize_value(info, normalized[dim["name"]], for_optimizer=True)
            if dim["type"] in ("integer", "float"):
                lo = dim["min"]
                hi = dim["max"]
                denom = hi - lo if hi != lo else 1.0
                values.append((float(value) - lo) / denom)
            else:
                categories = dim["categories"]
                idx = categories.index(value) if value in categories else 0
                denom = max(1, len(categories) - 1)
                values.append(idx / denom)
        return values
