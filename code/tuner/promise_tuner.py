import time
import random
import numpy as np
import pandas as pd
from itertools import product
import copy
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import lingam
from causallearn.search.ConstraintBased.FCI import fci

from workload import WorkloadController
from utils.logger import Logger
from tqdm import tqdm
from systems.factory import create_target_system
from utils.knob_space_utils import build_categorical_values


class PromiseTuner:
    def __init__(self, args_db, args_workload, args_tune, run):
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
        self.log_file = 'PromiseTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        self.initial_size = 10
        self.l_value = 10
        self.k_percent = 0.1
        self.consumed_cost = 0
        self.consumed_iters = 0
        self.evaluated_data = []  # [(config, perf, cost)]
        self.evaluated_configs = set()  # Set of (config tuple, fidelity tuple)
        self.target_system = create_target_system(self.args_db)

        self.workload_controller = WorkloadController(args_db, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)
        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="iter")

    def tune_promise(self):
        start_time = time.time()
        hf_factors = self.workload_controller.get_default_fidelity_factors()
        self.promise_search_config(self.total_budget, hf_factors)
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time, self.log_path)
        self.pbar.close()

    def promise_search_config(self, budget, fidelity):
        init_configs = self.sample_random_configs(self.initial_size, fidelity, self.target_system.knobs_info)
        self.evaluate_and_record(init_configs, fidelity)

        while self.consumed_iters < budget:
            rules = self.extract_rules_from_rf()
            rule_features, labels = self.featurize_rules(rules)
            purified_rules = self.purify_rules_by_causality(rule_features, labels, rules)
            candidate_pool = self.sample_candidates_with_rules(purified_rules, fidelity)
            best_config = self.select_by_ei(candidate_pool, fidelity, self.target_system.knobs_info)
            self.evaluate_and_record([best_config], fidelity)

    def sample_random_configs(self, sampling_size, fidelity, knobs_info):
        init_configs = []
        seen_configs = set()
        while len(init_configs) < sampling_size:
            config = {}
            for knob_name, knob_info in knobs_info.items():
                knob_type = str(knob_info.get('type', '')).lower()
                if knob_type == 'integer':
                    random_value = random.randint(knob_info['min'], knob_info['max'])
                    config[knob_name] = random_value
                elif knob_type == 'float':
                    random_value = random.uniform(knob_info['min'], knob_info['max'])
                    config[knob_name] = random_value
                elif knob_type in ('enum', 'boolean'):
                    possible_value = self._categorical_values(knob_info)
                    index = random.randint(0, len(possible_value) - 1)
                    config[knob_name] = possible_value[index]
            config_tuple = tuple(config.items())
            if config_tuple not in seen_configs:
                init_configs.append(config)
                seen_configs.add(config_tuple)
        return init_configs

    def evaluate_and_record(self, configs, fidelity, stage_budget=None):
        evaluated_configs, cost = self.evaluate_configs(configs, fidelity, stage_budget)
        for config, perf, evaluated_cost in evaluated_configs:
            key = (tuple(sorted(config.items())), tuple(sorted(fidelity.items())))
            self.evaluated_configs.add(key)
            self.evaluated_data.append((config, perf, evaluated_cost))
        return evaluated_configs, cost

    def evaluate_configs(self, configs, factors, stage_budget=None):
        evaluated_configs = []
        cost_configs = 0
        current_loop_consumption = 0
        for config in configs:
            self.target_system.set_db_knob(config)
            num_iter = 1
            total_perf = total_prepare_time = total_run_time = total_clean_time = total_evaluated_cost = 0
            total_latency = total_throughput = total_qps = 0
            for _ in range(num_iter):
                start = time.time()
                try:
                    print(f"Execute workload: {factors}")
                    latency, throughput, qps, prepare_time, run_time, clean_time = self.workload_controller.run_workload(factors)

                except Exception as e:
                    print(f"Workload execution failed: {e}")
                    latency = throughput = qps = prepare_time = run_time = clean_time = 0
                end = time.time()
                evaluated_cost = end - start
                total_perf += self._extract_objective_value(latency, throughput, qps, run_time)
                total_latency += latency
                total_throughput += throughput
                total_qps += qps
                total_prepare_time += prepare_time
                total_run_time += run_time
                total_clean_time += clean_time
                total_evaluated_cost += evaluated_cost

            perf = total_perf / num_iter
            avg_latency = total_latency / num_iter
            avg_throughput = total_throughput / num_iter
            avg_qps = total_qps / num_iter
            prepare_time = total_prepare_time / num_iter
            run_time = total_run_time / num_iter
            clean_time = total_clean_time / num_iter
            evaluated_cost = total_evaluated_cost / num_iter

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
            self.logger.logging_cyber_twin(config, perf, evaluated_cost, factors, prepare_time, run_time, clean_time,
                                           self.cyber_twin_path, self.cyber_twin_file)

            if self.consumed_iters >= self.total_budget:
                break
            if stage_budget is not None and current_loop_consumption >= stage_budget:
                print("stage budge exhausted.")
                break
        return evaluated_configs, cost_configs

    def _extract_objective_value(self, latency, throughput, qps, run_time):
        objective = str(self.optimize_objective).strip().lower()
        if objective in {'throughput', 'rps', 'encodefps'}:
            return throughput
        if objective == 'qps':
            return qps
        if objective in {'latency', 'encode_time'}:
            return latency
        if objective == 'run_time':
            return run_time
        raise ValueError(f"Unsupported optimization objective: {self.optimize_objective}")


    def extract_rules_from_rf(self):
        configs = [c for c, _, _ in self.evaluated_data]
        y = [p for _, p, _ in self.evaluated_data]
        feature_names = list(self.target_system.knobs_info.keys())
        rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=self.l_value)
        x = self.preprocess_configs(configs)
        rf.fit(x, y)
        all_rules = []
        for tree in rf.estimators_:
            rules = self.get_tree_paths(tree, feature_names)
            all_rules.extend([r for r in rules if r not in all_rules])
        return all_rules

    def get_tree_paths(self, tree, feature_names):
        paths = []
        stack = [(0, [])]
        while stack:
            node_id, path = stack.pop()
            feature_index = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]
            if feature_index != -2:
                name = feature_names[feature_index]
                stack.append((tree.tree_.children_left[node_id], path + [(name, threshold, 'L')]))
                stack.append((tree.tree_.children_right[node_id], path + [(name, threshold, 'R')]))
            else:
                paths.append(path)
        return paths

    def preprocess_configs(self, configs):
        knobs_info = self.target_system.knobs_info
        processed = []
        for config in configs:
            row = []
            for k in knobs_info:
                val = config[k]
                if str(knobs_info[k].get('type', '')).lower() in ('enum', 'boolean'):
                    row.append(self._categorical_index(knobs_info[k], val))
                else:
                    row.append(val)
            processed.append(row)
        return processed

    def featurize_rules(self, rules):
        configs = [c for c, _, _ in self.evaluated_data]
        feature_names = list(self.target_system.knobs_info.keys())
        feature_matrix = []
        for config in configs:
            row = []
            for rule in rules:
                row.append(1 if self.config_in_rule(config, rule, feature_names) else 0)
            feature_matrix.append(row)
        labels = [p for _, p, _ in self.evaluated_data]
        return feature_matrix, labels

    def config_in_rule(self, config, rule, feature_names):
        knobs_info = self.target_system.knobs_info
        for key, threshold, direction in rule:
            val = config[key]
            if str(knobs_info[key].get('type', '')).lower() in ('enum', 'boolean'):
                val = self._categorical_index(knobs_info[key], val)

            if direction == 'L' and not val <= threshold:
                return False
            if direction == 'R' and not val > threshold:
                return False
        return True

    def purify_rules_by_causality(self, X, y, rules):
        """
        Perform causal filtering using DirectLiNGAM on rule-feature matrix.
        Rules with negative ACE (average causal effect) on performance are retained.
        """
        # 1. 构造 DataFrame：每列是一个 rule_feature，最后一列是 performance
        df = pd.DataFrame(X, columns=[f'rule_{i}' for i in range(len(rules))])
        df['perf'] = y
        data = df.to_numpy()
        num_features = data.shape[1]

        # 2. 初始化因果结构 prior_knowledge
        try:
            graph, _ = fci(data, show_progress=False, verbose=False)
            adjacency_matrix = self.get_adjacency_matrix(graph)
            if adjacency_matrix.shape != (num_features, num_features):
                raise ValueError("Adjacency matrix shape mismatch.")
            adjacency_matrix = adjacency_matrix.T
            adjacency_matrix[:, -1] = 0  # 保证 performance 没有 outgoing edge
        except Exception as e:
            print(f"[DEBUG] FCI failed or matrix invalid: {e}")
            adjacency_matrix = np.zeros((num_features, num_features), dtype=int)

        # 3. 构造 prior_knowledge
        prior_knowledge = adjacency_matrix.copy()
        prior_knowledge[-1, :-1] = 1  # perf 可以被所有 rule 影响
        prior_knowledge[-1, -1] = -1  # perf 不影响任何其他变量

        # 4. 利用 DirectLiNGAM 拟合因果模型
        purified = []
        try:
            model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
            model.fit(data)
            target_idx = num_features - 1
            print(f"[DEBUG] Num samples: {len(y)}, Num rules: {len(rules)}")
            for i in range(len(rules)):
                ace = model.estimate_total_effect(data, i, target_idx)
                print(f"[DEBUG] ACE[{i}]: {ace:.4f}")
                if ace < 0:
                    purified.append(rules[i])
        except Exception as e:
            print(f"[DEBUG] Causality analysis failed: {e}")

        return purified

    def get_adjacency_matrix(self, graph):
        nodes = graph.get_nodes()
        node_count = len(nodes)
        matrix = np.zeros((node_count, node_count), dtype=int)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        for edge in graph.get_graph_edges():
            i = node_index[edge.node1]
            j = node_index[edge.node2]
            matrix[i, j] = 1
        return matrix

    def sample_candidates_with_rules(self, rules, fidelity):
        knobs_info = self.target_system.knobs_info
        feature_names = list(knobs_info.keys())
        all_candidates = []

        configs = [c for c, _, _ in self.evaluated_data]
        perfs = [p for _, p, _ in self.evaluated_data]
        X = self.preprocess_configs(configs)
        model = RandomForestRegressor()
        model.fit(X, perfs)
        estimators = model.estimators_
        eta = min(perfs) if self.optimize_objective in ['latency', 'run_time'] else max(perfs)

        # fallback if no rule
        if not rules:
            candidates, _ = self.ei_guided_sampling_from_domain(knobs_info, estimators, eta, fidelity)
            return candidates

        for rule in rules:
            modified_knobs_info = self.modify_knob_info_by_rule(rule, knobs_info)
            if modified_knobs_info is None:
                continue
            candidates, _ = self.ei_guided_sampling_from_domain(modified_knobs_info, estimators, eta, fidelity)
            all_candidates.extend(candidates)

        return all_candidates

    def modify_knob_info_by_rule(self, rule, original_knobs_info):
        knobs_info = {k: copy.deepcopy(v) for k, v in original_knobs_info.items()}
        if len(rule) > 1:
            print(f"[DEBUG] Applying rule: {rule} to knobs_info")
        for r_knob, threshold, direction in rule:
            if r_knob not in knobs_info:
                continue
            info = knobs_info[r_knob]

            # 修复 threshold 是 numpy 类型
            if isinstance(threshold, (np.ndarray, list)):
                threshold = threshold[0]

            knob_type = str(info.get('type', '')).lower()
            if knob_type == 'integer':
                threshold = int(threshold)
                if direction == 'L':
                    info['max'] = min(info['max'], threshold)
                elif direction == 'R':
                    info['min'] = max(info['min'], threshold + 1)

            elif knob_type == 'float':
                threshold = float(threshold)
                if direction == 'L':
                    info['max'] = min(info['max'], threshold)
                elif direction == 'R':
                    info['min'] = max(info['min'], threshold)

            elif knob_type in ('enum', 'boolean'):
                enum_values = self._categorical_values(info)
                try:
                    threshold_index = int(threshold)
                    if threshold_index < 0 or threshold_index >= len(enum_values):
                        continue
                    threshold_value = enum_values[threshold_index]
                except:
                    continue

                if direction == 'L':
                    info['enum_values'] = [v for v in enum_values if enum_values.index(v) <= threshold_index]
                elif direction == 'R':
                    info['enum_values'] = [v for v in enum_values if enum_values.index(v) > threshold_index]

                if not info['enum_values']:
                    return None

        return knobs_info

    def ei_guided_sampling_from_domain(self, knobs_info, estimators, eta, fidelity, max_iterations=10000, stop_threshold=0.01):
        feature_names = list(knobs_info.keys())
        values, configs, best = [], [], float('-inf')

        for i in range(max_iterations):
            sampled_config = self.sample_random_configs(1, fidelity, knobs_info)[0]
            x = self.preprocess_configs([sampled_config])[0]
            preds = [e.predict([x])[0] for e in estimators]
            mean = np.mean(preds)
            std = np.std(preds)
            if self.optimize_objective in ['latency', 'run_time']:  # minimization
                z = (eta - mean) / std if std > 0 else 0
                ei = (eta - mean) * norm.cdf(z) + std * norm.pdf(z) if std > 0 else 0
            else:  # throughput: maximization
                z = (mean - eta) / std if std > 0 else 0
                ei = (mean - eta) * norm.cdf(z) + std * norm.pdf(z) if std > 0 else 0

            values.append(ei)
            configs.append(sampled_config)
            best = max(best, ei)

            if i > 0:  # kde requires at least two points
                try:
                    kde = stats.gaussian_kde(values)
                    if 1 - kde.integrate_box_1d(-np.inf, best) < stop_threshold:
                        break
                except Exception as e:
                    break

        return configs, best

    def select_by_ei(self, candidates, fidelity, knobs_info):
        if not candidates:
            return random.choice(self.sample_random_configs(1, fidelity, knobs_info))
        train_x = self.preprocess_configs([c for c, _, _ in self.evaluated_data])
        train_y = [p for _, p, _ in self.evaluated_data]
        model = RandomForestRegressor()
        model.fit(train_x, train_y)

        test_x = self.preprocess_configs(candidates)
        preds = np.array([e.predict(test_x) for e in model.estimators_]).T
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        std[std == 0] = 1e-8
        best_seen = min(train_y) if self.optimize_objective in ['latency', 'run_time'] else max(train_y)
        if self.optimize_objective in ['latency', 'run_time']:  # minimization
            z = (best_seen - mean) / std
            ei = (best_seen - mean) * norm.cdf(z) + std * norm.pdf(z)
        else:  # maximization
            z = (mean - best_seen) / std
            ei = (mean - best_seen) * norm.cdf(z) + std * norm.pdf(z)

        return candidates[np.argmax(ei)]

    @staticmethod
    def _categorical_values(knob_info):
        return build_categorical_values(knob_info)

    @classmethod
    def _categorical_index(cls, knob_info, value):
        categories = cls._categorical_values(knob_info)
        return categories.index(value)

    def mock_run_workload(self, factors):
        """
        Mock function to simulate workload execution for debugging purposes.
        """
        # 模拟 workload 运行时间
        time.sleep(0.1)  # 模拟轻量执行

        # 随机生成指标值（符合一般直觉）
        latency = round(random.uniform(10, 200), 2)          # ms
        throughput = round(random.uniform(100, 10000), 2)    # ops/sec
        prepare_time = round(random.uniform(0.1, 1.0), 2)     # seconds
        run_time = round(random.uniform(100, 200), 2)         # seconds
        clean_time = round(random.uniform(0.1, 0.8), 2)       # seconds

        return latency, throughput, prepare_time, run_time, clean_time
