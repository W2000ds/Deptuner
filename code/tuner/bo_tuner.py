from skopt import Optimizer
from skopt.space import Categorical, Integer, Real

from tuner.depaware_tuner_base import DependencyAwareTunerBase
from utils.knob_space_utils import build_categorical_values, canonicalize_value, sanitize_config_for_optimizer


class BOTuner(DependencyAwareTunerBase):
    def __init__(self, args_db, args_workload, args_tune, run):
        super().__init__(args_db, args_workload, args_tune, run, "bo", "BOTuner_results.csv")
        self.dimensions, self.dimension_names = self._build_dimensions()
        self.optimizer = Optimizer(
            dimensions=self.dimensions,
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=int(args_tune.get("bo_n_initial_points", 20)),
            random_state=int(run),
        )
        self.reject_streak = 0
        self.reject_retry_limit = int(args_tune.get("dep_reject_retry_limit", 32))
        self.candidate_batch_size = int(
            args_tune.get("bo_candidate_batch_size", args_tune.get("dep_candidate_batch_size", 16))
        )

    def tune_bo(self):
        self.tune_single_fidelity(self.bo_search_config)

    def bo_search_config(self, budget, fidelity, start_consumed=0):
        consumed_iters = int(start_consumed)
        while consumed_iters < budget:
            x = self._select_candidate(fidelity)
            config = self._decode(x)

            violated_ids = []
            penalty = 0.0
            if self.dependency_aware:
                reject, _ = self.dep_manager.should_reject_hard(config)
                if reject:
                    self.reject_streak += 1
                    if self.reject_streak >= self.reject_retry_limit:
                        fallback = self._sample_feasible_fallback(fidelity)
                        if fallback is not None:
                            x, config = fallback
                            self.reject_streak = 0
                            reject, _ = self.dep_manager.should_reject_hard(config)
                    if reject:
                        continue
                else:
                    self.reject_streak = 0
                penalty, violated_ids, _ = self.dep_manager.penalty(config)
            else:
                self.reject_streak = 0

            if self.config_key(config, fidelity) in self.evaluated_configs:
                self.reject_streak += 1
                continue

            evaluated_config, _ = self.evaluate_configs([config], fidelity)
            if not evaluated_config:
                self.reject_streak += 1
                continue

            raw_perf = evaluated_config[0][1]
            perf = self.dependency_adjusted_perf(config, raw_perf, violated_ids, penalty, fidelity)
            self.optimizer.tell(x, self.objective_value(perf))
            self.evaluated_configs.add(self.config_key(config, fidelity))
            self.reject_streak = 0
            consumed_iters += len(evaluated_config)

    def observe_probe_config(self, config, adjusted_perf):
        normalized = sanitize_config_for_optimizer(self.target_system.knobs_info, config)
        x = [normalized[name] for name in self.dimension_names]
        self.optimizer.tell(x, self.objective_value(adjusted_perf))

    def _select_candidate(self, fidelity):
        if not self.dependency_aware:
            return self.optimizer.ask()

        batch_size = max(1, self.candidate_batch_size)
        candidates = self.optimizer.ask(n_points=batch_size)
        for x in candidates:
            config = self._decode(x)
            if self.config_key(config, fidelity) in self.evaluated_configs:
                continue
            reject, _ = self.dep_manager.should_reject_hard(config)
            if reject:
                continue
            return x

        return self.optimizer.ask()

    def _sample_feasible_fallback(self, fidelity):
        for _ in range(max(8, self.reject_retry_limit)):
            x = self.optimizer.space.rvs(n_samples=1)[0]
            config = self._decode(x)
            if self.config_key(config, fidelity) in self.evaluated_configs:
                continue
            reject, _ = self.dep_manager.should_reject_hard(config)
            if reject:
                continue
            return x, config
        return None

    def _build_dimensions(self):
        dimensions = []
        names = []
        for name, info in self.target_system.knobs_info.items():
            ktype = str(info.get("type", "")).lower()
            if ktype == "integer":
                dimensions.append(Integer(int(info["min"]), int(info["max"]), name=name))
            elif ktype == "float":
                dimensions.append(Real(float(info["min"]), float(info["max"]), name=name))
            elif ktype in ("enum", "boolean"):
                categories = build_categorical_values(info)
                dimensions.append(Categorical(categories, name=name))
            names.append(name)
        return dimensions, names

    def _decode(self, x):
        config = {}
        for name, value in zip(self.dimension_names, x):
            info = self.target_system.knobs_info[name]
            if str(info.get("type", "")).lower() == "integer":
                config[name] = canonicalize_value(info, value, for_optimizer=True)
            elif str(info.get("type", "")).lower() == "float":
                config[name] = canonicalize_value(info, value, for_optimizer=True)
            else:
                config[name] = canonicalize_value(info, value, for_optimizer=True)
        return self._apply_bo_safety_guards(config)

    def _apply_bo_safety_guards(self, config):
        guarded = dict(config)
        if self.sys_name == "httpd":
            guarded = self._guard_httpd_config(guarded)
        elif self.sys_name == "mysql":
            guarded = self._guard_mysql_config(guarded)
        return guarded

    def _guard_httpd_config(self, config):
        # Very small process / memory rlimits can leave Apache unable to become ready.
        floors = {
            "RLimitCPU": 30,
            "RLimitMEM": 256 * 1024 * 1024,
            "RLimitNPROC": 256,
        }
        for knob, floor in floors.items():
            if knob not in config:
                continue
            value = config[knob]
            try:
                numeric_value = int(float(value))
            except Exception:
                continue
            if numeric_value < floor:
                default_value = self.target_system.knobs_info.get(knob, {}).get("default", "unset")
                config[knob] = default_value
        return config

    def _guard_mysql_config(self, config):
        # The sort-buffer workload is extremely sensitive to tiny temp-table limits and
        # oversized max_sort_length relative to sort_buffer_size. Clamp BO proposals to
        # a stable subspace before evaluating them.
        sort_buffer_size = self._coerce_int(config.get("sort_buffer_size"), fallback=262144)
        sort_buffer_size = max(sort_buffer_size, 2 * 1024 * 1024)
        config["sort_buffer_size"] = sort_buffer_size

        tmp_floor = 16 * 1024 * 1024
        tmp_table_size = self._coerce_int(config.get("tmp_table_size"), fallback=tmp_floor)
        max_heap_table_size = self._coerce_int(config.get("max_heap_table_size"), fallback=tmp_floor)
        temp_limit = max(tmp_floor, tmp_table_size, max_heap_table_size)
        config["tmp_table_size"] = temp_limit
        config["max_heap_table_size"] = temp_limit

        join_buffer_size = self._coerce_int(config.get("join_buffer_size"), fallback=262144)
        config["join_buffer_size"] = max(join_buffer_size, 262144)

        max_sort_length = self._coerce_int(config.get("max_sort_length"), fallback=1024)
        safe_max_sort_length = max(4096, sort_buffer_size // 32)
        config["max_sort_length"] = min(max_sort_length, safe_max_sort_length)
        return config

    @staticmethod
    def _coerce_int(value, fallback):
        try:
            return int(float(value))
        except Exception:
            return int(fallback)
