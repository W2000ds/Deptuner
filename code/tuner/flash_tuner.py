
import random
import time
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from sklearn.tree import DecisionTreeRegressor
from systems.factory import create_target_system
from utils.knob_space_utils import build_categorical_values


class FLASHTuner:
    def __init__(self, args_db, args_workload, args_tune, run):
        super(FLASHTuner, self).__init__()
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
        self.log_file = 'FLASHTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Parameters Settings for Algorithm: Flash for config space
        self.initial_size = 30
        self.sampling_size = 1000
        self.evaluated_configs = set()
        self.consumed_cost = 0
        self.consumed_iters = 0

        self.target_system = create_target_system(self.args_db)

        # Workload Controller; Logger;
        self.workload_controller = WorkloadController(args_db, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)
        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="iter")




    def tune_flash(self):
        start_time = time.time()

        if self.fidelity_type == 'single_fidelity':
            hf_factors = self.workload_controller.get_default_fidelity_factors()
            self.flash_search_config(self.total_budget, hf_factors)
        elif self.fidelity_type == 'multi_fidelity':
            pass

        end_time = time.time()
        runtime = end_time - start_time

        # Record runtime using the Logger class
        self.logger.store_runtime_to_csv(runtime, self.log_path)
        self.pbar.close()

    def flash_search_config(self, budget, fidelity, init_configs=None, kd_corr=1, fidelity_id=0,
                            evaluated_filtered_configs=None):
        """
        :param lf_filtered_configs:
        :param fidelity_id:
        :param kd_corr:
        :param init_configs:
        :param budget:
        :param fidelity:
        :return:
        """

        consumed_iters = 0
        if init_configs is None:
            init_configs = self.sampling_configs(self.initial_size)
        evaluated_configs, init_eval_count = self.evaluate_configs(init_configs, fidelity)
        consumed_iters += init_eval_count

        # prepared for extension to multi-fidelity
        if evaluated_filtered_configs:
            # Combine and sort the results from both evaluations
            evaluated_configs = evaluated_configs + evaluated_filtered_configs
            if self.optimize_objective == 'throughput':
                evaluated_configs.sort(key=lambda x: x[1], reverse=True)
            elif self.optimize_objective == 'latency':
                evaluated_configs.sort(key=lambda x: x[1])

        evaluated_configs = evaluated_configs[:self.initial_size]
        # Record the evaluated configs / with fidelity as part of the key
        for config, _, _ in evaluated_configs:
            self.evaluated_configs.add((tuple(sorted(config.items())), tuple(sorted(fidelity.items()))))

        while consumed_iters < budget:

            # Train a CART: regression decision tree model
            model = DecisionTreeRegressor()
            configs = [config for config, _, _ in evaluated_configs]
            train_x = self.preprocess_configs_with_knobs_info(configs, self.target_system.knobs_info)
            # train_x = [list(config.values()) for config, _, _ in evaluated_configs]
            train_y = [perf for _, perf, _ in evaluated_configs]
            model.fit(train_x, train_y)

            sampled_configs = self.sampling_configs(self.sampling_size)
            # Filter those configs that haven't been evaluated in the past
            unevaluated_configs = [config for config in sampled_configs if (
                tuple(sorted(config.items())), tuple(sorted(fidelity.items()))
            ) not in self.evaluated_configs]

            # Transform the unevaluated configs at two-dimensional list, get the value
            # As some of the configs that may take values as strings, preprocess is needed
            test_x = self.preprocess_configs_with_knobs_info(unevaluated_configs, self.target_system.knobs_info)
            # test_x = [list(config.values()) for config in unevaluated_configs]

            # Predict the performance for given configs by trained model
            predicted_performances = model.predict(test_x)

            if self._is_maximize_objective():
                best_config_index = predicted_performances.argmax()
            else:
                best_config_index = predicted_performances.argmin()

            best_config = unevaluated_configs[best_config_index]

            # Implement true system measurement for estimated best config
            evaluated_best_config, best_eval_count = self.evaluate_configs([best_config], fidelity)
            consumed_iters += best_eval_count

            # Update the evaluated config (training data)
            evaluated_configs += evaluated_best_config
            self.evaluated_configs.add((tuple(sorted(best_config.items())), tuple(sorted(fidelity.items()))))

        return evaluated_configs

    def sampling_configs(self, sampling_size):
        """Random sampling"""
        init_configs = []
        seen_configs = set()
        while len(init_configs) < sampling_size:
            config = {}
            for knob_name, knob_info in self.target_system.knobs_info.items():
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

    def preprocess_configs_with_knobs_info(self, configs, knobs_info):
        """
        Preprocessing configs dynamically according to knobs_info
        :param configs: config list，dict{config_name: config_value}
        :param knobs_info: detailed info for config
        :return: transformed two-dimensional list
        """
        processed_configs = []
        for config in configs:
            processed_config = []
            for key, value in config.items():
                if key in knobs_info and str(knobs_info[key].get("type", "")).lower() in ("enum", "boolean"):
                    # extract the index as it's value for the purpose of training model
                    enum_values = self._categorical_values(knobs_info[key])
                    processed_config.append(enum_values.index(value))
                else:
                    processed_config.append(value)
            processed_configs.append(processed_config)
        return processed_configs

    def evaluate_configs(self, configs, factors, stage_budget=None):
        evaluated_configs = []
        evaluated_count = 0
        current_loop_evals = 0
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
            current_loop_evals += 1
            evaluated_count += 1
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
            if stage_budget is not None and current_loop_evals >= stage_budget:
                print("stage budge exhausted.")
                break
        return evaluated_configs, evaluated_count

    def _is_maximize_objective(self):
        return str(self.optimize_objective).strip().lower() in {
            "throughput",
            "qps",
            "rps",
            "encodefps",
        }

    def _extract_objective_value(self, latency, throughput, qps, run_time):
        objective = str(self.optimize_objective).strip().lower()
        if objective in {"throughput", "rps", "encodefps"}:
            return throughput
        if objective == "qps":
            return qps
        if objective in {"latency", "encode_time"}:
            return latency
        if objective == "run_time":
            return run_time
        raise ValueError(f"Unsupported optimization objective: {self.optimize_objective}")

    @staticmethod
    def _categorical_values(knob_info):
        return build_categorical_values(knob_info)
