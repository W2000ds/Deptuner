import random
import sys
import time
import os
import numpy as np
from pyDOE import lhs
from tqdm import tqdm
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from systems.mysqldb import MysqlDB
from systems.postgresqldb import PostgresqlDB
from workload import WorkloadController
from utils.logger import Logger
from utils.multfidelity_optimizer import MultiFidelityOptimizer
from utils.config_utils import ConfigUtils
from utils.path_layout import default_online_rule_file
from dependency.dependency_manager import DependencyManager
from dependency.evidence_context import build_context_parts
from dependency.evidence_extractor import RuntimeEvidenceExtractor


class BayesTuner:
    def __init__(self, args_db, args_workload, args_tune, run):
        super(BayesTuner, self).__init__()
        self.max_iter = int(args_tune['max_iter'])
        self.total_budget = int(args_tune['total_budget'])
        self.args_db = args_db
        self.args_workload = args_workload
        self.args_tune = args_tune
        self.workload_bench = args_workload["workload_bench"]
        self.tuning_method = args_tune['tuning_method']
        self.fidelity_type = args_tune['fidelity_type']
        self.fidelity_metric = args_tune['fidelity_metric']
        self.optimize_objective = args_tune['optimize_objective']
        self.sys_name = self.args_db['db']
        self.log_path = f'experimental_results/{self.sys_name}/{self.workload_bench}/{self.tuning_method}/run_{run}_{self.tuning_method}_{self.fidelity_type}'
        self.log_file = 'BayesTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'
        self.budget_4_fidelity_search = 0

        # Parameters Settings for BO
        self.initial_size = 30
        self.sampling_size = 1000
        self.evaluated_configs = set()
        self.consumed_cost = 0

        # Target system
        if self.sys_name == 'mysql':
            self.target_system = MysqlDB(self.args_db)
        elif self.sys_name == 'postgresql':
            self.target_system = PostgresqlDB(self.args_db)

        # Workload Controller; Logger;
        self.workload_controller = WorkloadController(args_db, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)
        self.multi_fidelity_optimizer = MultiFidelityOptimizer(self.workload_controller,
                                                               self.workload_controller.fidelity_factors_info,
                                                               self.evaluate_configs, self.target_system,
                                                               self.optimize_objective,
                                                               self.max_iter, self.fidelity_metric)

        self.hf_evaluated_configs = []
        self.PROB_HIGH_FIDELITY_TRIGGER = 0.1
        self.dep_iter = 0
        self.dep_degrade_ratio = float(args_tune.get("dep_degrade_ratio", 0.10))
        self.dep_ema_alpha = float(args_tune.get("dep_ema_alpha", 0.30))
        self.perf_baseline = None

        self.dependency_aware = self._to_bool(args_tune.get("dependency_aware", "false"))
        self.dep_manager = None
        self.evidence_extractor = None
        if self.dependency_aware:
            rule_file = str(args_tune.get("dependency_rule_file", "")).strip()
            if not rule_file:
                rule_file = default_online_rule_file(self.sys_name)
            if os.path.exists(rule_file):
                self.dep_manager = DependencyManager.from_file(
                    rule_file,
                    lambda_e=float(args_tune.get("dep_lambda_e", 0.05)),
                    lambda_f=float(args_tune.get("dep_lambda_f", 0.20)),
                    hard_on_exist=float(args_tune.get("dep_hard_on_exist", 0.8)),
                    gray_budget_ratio=float(args_tune.get("dep_gray_budget_ratio", 0.1)),
                    exist_step=float(args_tune.get("dep_exist_step", 0.10)),
                    fail_step=float(args_tune.get("dep_fail_step", 0.10)),
                    fail_credit_mode=args_tune.get("dep_fail_credit_mode", "independent"),
                    use_p_exist=self._to_bool(args_tune.get("dep_use_p_exist", "true")),
                    use_p_fail=self._to_bool(args_tune.get("dep_use_p_fail", "true")),
                )
                if not self.dep_manager.enabled():
                    self.dep_manager = None
            else:
                print(f"[DependencyAware] Rule file not found: {rule_file}; dependency aware disabled.")
        self.dependency_aware = self.dep_manager is not None
        if self.dependency_aware:
            self.evidence_extractor = RuntimeEvidenceExtractor(
                self.workload_controller,
                self.target_system,
                self.dep_manager,
            )

        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")

    def tune_bayes(self):
        start_time = time.time()
        if self.fidelity_type == 'single_fidelity':
            hf_factors = self.workload_controller.get_default_fidelity_factors()
            self.bayes_search_config(self.total_budget, hf_factors)
        elif self.fidelity_type == 'multi_fidelity':
            
            print("Start running BO with Multi-fidelity...")
            pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost", file=sys.stdout)


            self.budget_4_fidelity_search = self.total_budget * 0.2
            sample_size = 10
            config_samples = ConfigUtils.sampling_configs_by_lhs(sample_size, self.target_system.knobs_info)
            hf_factors = self.workload_controller.get_default_fidelity_factors()
            hf_evaluated_samples, hf_evaluated_samples_cost = self.evaluate_configs(config_samples, hf_factors)
            self.hf_evaluated_configs.extend(hf_evaluated_samples)
            hf_perf = [perf for _, perf, _ in hf_evaluated_samples]
            hf_cost = [cost for _, _, cost in hf_evaluated_samples]
            pbar.update(hf_evaluated_samples_cost)
            cost_related_factors, cost_dva = self.multi_fidelity_optimizer.decision_variable_analysis(
                hf_evaluated_samples, hf_factors, 5, 5)
            self.logger.log_cost_related_factors(cost_related_factors, self.log_path)

            if not cost_related_factors:
                # Without any factors that can save cost
                print("No cost-related factors found, proceeding with single fidelity optimization.")
                self.bayes_search_config(self.total_budget - self.consumed_cost, hf_factors)
                return

            print("[1] Explore fidelity settings with high fidelity and low cost")
            # Fidelity Optimization and Fidelity Measurement
            optimized_fidelity_pop = self.multi_fidelity_optimizer.evolutionary_search_fidelity(hf_factors,
                                                                                                cost_related_factors,
                                                                                                config_samples, hf_perf,
                                                                                                self.log_path, cost_dva,
                                                                                                self.budget_4_fidelity_search - self.consumed_cost)

            print(f"Consumed cost for fidelity settings identify: {self.consumed_cost}")
            print(f"The rest of budget: {self.total_budget - self.consumed_cost}")

            # Fidelity Management for sequential usage.
            budget_4_hfbo = self.total_budget * 0.4  # preserve 30% budget for the last high-fidelity stage
            if self.total_budget - self.consumed_cost < budget_4_hfbo:
                raise ValueError("Total budget is not enough to ensure the final stage evaluation at highest fidelity.")

            print("[2] Divide non-dominated fidelity settings")
            selected_fidelities = self.multi_fidelity_optimizer.select_fidelity_by_knee_point(optimized_fidelity_pop,
                                                                                              self.log_path)
            if selected_fidelities is None:
                print(f"Without suitable fidelity setting, using high-fidelity search: {self.total_budget - self.consumed_cost}")
                self.bayes_search_config(self.total_budget - self.consumed_cost, hf_factors)
                end_time = time.time()
                runtime = end_time - start_time
                # Record runtime using the Logger class
                self.logger.store_runtime_to_csv(runtime, self.log_path)
                return


            budget_4_lfbo = self.total_budget - self.consumed_cost - budget_4_hfbo
            lf_factors, lf_corr, lf_cost = selected_fidelities[0]
            fidelity_id = 1
            lfbo_evaluated_configs = self.bayes_search_config(budget_4_lfbo, lf_factors, kd_corr=lf_corr, fidelity_id=fidelity_id)
            self.logger.store_config_pop_to_csv(lfbo_evaluated_configs, 0, lf_factors, lf_corr,
                                                fidelity_id, self.log_path)

            promising_configs = ConfigUtils.get_top_k_configs(lfbo_evaluated_configs, k=self.initial_size, optimize_objective=self.optimize_objective)
            promising_configs = [config for config in promising_configs if config not in [c for c, _, _ in self.hf_evaluated_configs]]


            budget_4_hfbo = self.total_budget - self.consumed_cost
            print(f"Final stage (full fidelity) budget: {budget_4_hfbo}")
            self.bayes_search_config(budget_4_hfbo, hf_factors, init_configs=promising_configs, hf_evaluated_configs=self.hf_evaluated_configs)
            
        end_time = time.time()
        runtime = end_time - start_time
        self.logger.store_runtime_to_csv(runtime, self.log_path)
        self.pbar.close()

    def bayes_search_config(self, budget, fidelity, init_configs=None, kd_corr=1, fidelity_id=0, hf_evaluated_configs=None):

        pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")
        consumed_cost = 0

        if init_configs is None:
            init_configs = ConfigUtils.sampling_configs_by_rs(self.initial_size, self.target_system.knobs_info)
        evaluated_configs, cost_init_configs = self.evaluate_configs(init_configs, fidelity)
        if not evaluated_configs:
            print("No valid initial configs evaluated. Ending search.")
            pbar.close()
            return []
        consumed_cost += cost_init_configs
        pbar.update(cost_init_configs)

        if hf_evaluated_configs and fidelity_id == 0:
            evaluated_configs += hf_evaluated_configs
            

        for config, _, _ in evaluated_configs:
            self.evaluated_configs.add((tuple(sorted(config.items())), tuple(sorted(fidelity.items()))))

        while consumed_cost < budget:

            model = RandomForestRegressor(n_estimators=10)  # forest with 10 trees.
            configs = [config for config, _, _ in evaluated_configs]
            train_x = ConfigUtils.preprocess_configs_with_knobs_info(configs, self.target_system.knobs_info)
            train_y = [perf for _, perf, _ in evaluated_configs]
            model.fit(train_x, train_y)

            sampled_configs = ConfigUtils.sampling_configs_by_rs(self.sampling_size, self.target_system.knobs_info)
            unevaluated_configs = [config for config in sampled_configs if (
                tuple(sorted(config.items())), tuple(sorted(fidelity.items()))
            ) not in self.evaluated_configs]

            test_x = ConfigUtils.preprocess_configs_with_knobs_info(unevaluated_configs, self.target_system.knobs_info)
            preds = model.predict(test_x)
            stds = np.std([tree.predict(test_x) for tree in model.estimators_], axis=0)

            y_best = max(train_y) if self.optimize_objective in ['throughput', 'RPS'] else min(train_y)
            ei_values = self.expected_improvement(np.array(preds), np.array(stds), y_best,
                                                  maximize=self.optimize_objective in ['throughput', 'RPS'])

            if len(ei_values) == 0 or np.all(np.isnan(ei_values)):
                print("No valid EI values. Ending search.")
                break

            acq_values = np.array(ei_values, copy=True)
            if self.dependency_aware:
                for idx, cfg in enumerate(unevaluated_configs):
                    penalty, _, _ = self.dep_manager.penalty(cfg)
                    acq_values[idx] = acq_values[idx] - penalty

            candidate_indices = np.argsort(-acq_values)
            best_config = None
            for idx in candidate_indices:
                cfg = unevaluated_configs[int(idx)]
                if not self.dependency_aware:
                    best_config = cfg
                    break
                reject, _ = self.dep_manager.should_reject_hard(cfg)
                if not reject:
                    best_config = cfg
                    break

            if best_config is None:
                print("No selectable candidate after hard-constraint filtering. Ending search.")
                break

            evaluated_best_config, cost_best_config = self.evaluate_configs([best_config], fidelity)
            if not evaluated_best_config:
                continue
            consumed_cost += cost_best_config
            pbar.update(cost_best_config)

            evaluated_configs += evaluated_best_config
            self.evaluated_configs.add((tuple(sorted(best_config.items())), tuple(sorted(fidelity.items()))))

            if fidelity_id != 0 and random.random() <= self.PROB_HIGH_FIDELITY_TRIGGER:

                current_best_config = ConfigUtils.get_top_k_configs(evaluated_configs, 1, self.optimize_objective)
                current_config_tuple = tuple(sorted(current_best_config[0].items()))
                evaluated_set = set(tuple(sorted(config.items())) for config, _, _ in self.hf_evaluated_configs)
                if current_config_tuple not in evaluated_set:
                    hf_factors = self.workload_controller.get_default_fidelity_factors()
                    hf_evaluated_current_best_config, cost_best_config = self.evaluate_configs(current_best_config, hf_factors)
                    consumed_cost += cost_best_config
                    self.hf_evaluated_configs.extend(hf_evaluated_current_best_config)

        pbar.close()
        return evaluated_configs

    def expected_improvement(self, mu, sigma, y_best, maximize=True, xi=0.01):
        if not maximize:
            y_best = -y_best
            mu = -mu
        with np.errstate(divide='warn'):
            improvement = mu - y_best - xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def evaluate_configs(self, configs, factors, stage_budget=None):
        evaluated_configs = []
        cost_configs = 0
        current_loop_consumption = 0
        for config in configs:
            violated_ids = []
            if self.dependency_aware:
                reject, _ = self.dep_manager.should_reject_hard(config)
                if reject:
                    continue
                violated_ids = self.dep_manager.violated_rule_ids(config)

            knob_apply_ok = self.target_system.set_db_knob(config)
            num_iter = 1
            total_perf = total_prepare_time = total_run_time = total_clean_time = total_evaluated_cost = 0
            total_latency = total_throughput = total_qps = 0
            for _ in range(num_iter):
                start = time.time()
                try:
                    if not knob_apply_ok:
                        raise RuntimeError("knob apply failed")
                    print(f"Execute workload: {factors}")
                    latency, throughput, qps, prepare_time, run_time, clean_time = self.workload_controller.run_workload(
                        factors)
                except Exception as e:
                    print(f"Workload execution failed: {e}")
                    latency = throughput = qps = prepare_time = run_time = clean_time = 0
                end = time.time()
                evaluated_cost = end - start
                if self.optimize_objective == 'throughput':
                    total_perf += throughput
                elif self.optimize_objective == 'latency':
                    total_perf += latency
                elif self.optimize_objective == 'qps':
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
            # Prevent near-zero-cost failures from causing runaway iterations.
            evaluated_cost = max(evaluated_cost, 1.0)
            avg_latency = total_latency / num_iter
            avg_throughput = total_throughput / num_iter
            avg_qps = total_qps / num_iter

            self.pbar.update(evaluated_cost)
            print()
            print(f"[PERFORMANCE]: {self.optimize_objective}: {perf}")
            print("-------------------------------------------------")

            self.consumed_cost += evaluated_cost
            current_loop_consumption += evaluated_cost
            cost_configs += evaluated_cost
            evaluated_configs.append((config, perf, evaluated_cost))

            if self.dependency_aware:
                outcome = self.classify_outcome(perf)
                evidence_by_rule = self.extract_runtime_evidence(violated_ids, config=config, factors=factors)
                credits = self.dep_manager.update(violated_ids, outcome, evidence_by_rule=evidence_by_rule)
                self.dep_iter += 1
                dep_rows = self.dep_manager.snapshot_rows(self.dep_iter, violated_ids, credits, outcome)
                for row in dep_rows:
                    row["evidence_signal"] = evidence_by_rule.get(row["rule_id"], "none")
                    self.logger.logging_dependency_state(row, self.log_path, "dep_state.csv")

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
            self.logger.logging_cyber_twin(config, perf, evaluated_cost, factors, prepare_time, run_time, clean_time,
                                           self.cyber_twin_path, self.cyber_twin_file)

            if self.consumed_cost >= self.total_budget:
                break
            if stage_budget is not None and current_loop_consumption >= stage_budget:
                print("stage budge exhausted.")
                break
        return evaluated_configs, cost_configs

    def classify_outcome(self, perf):
        if perf <= 0:
            return "fail"
        if self.perf_baseline is None:
            self.perf_baseline = perf
            return "ok"

        outcome = "ok"
        if self.optimize_objective in ["throughput", "qps", "RPS"]:
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
        if not violated_ids or not self.dep_manager or self.evidence_extractor is None:
            return {}
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

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
