
import math

from tqdm import tqdm
import random
import time
from workload import WorkloadController
from utils.logger import Logger
from systems.factory import create_target_system
from utils.knob_space_utils import build_categorical_values


class BestConfigTuner:
    def __init__(self, args_db, args_workload, args_tune, run):
        super(BestConfigTuner, self).__init__()
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
        self.log_file = 'BestConfigTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Parameters Settings for Algorithm: Flash for config space
        self.max_rounds = 5
        self.sampling_size = 8
        self.evaluated_configs = set()
        self.consumed_cost = 0
        self.consumed_iters = 0
        self.target_system = create_target_system(self.args_db)

        # Workload Controller; Logger;
        self.workload_controller = WorkloadController(args_db, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)
        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="iter")

    def tune_bestconfig(self):
        start_time = time.time()

        if self.fidelity_type == 'single_fidelity':
            hf_factors = self.workload_controller.get_default_fidelity_factors()
            self.bestconfig_search_config(self.total_budget, hf_factors)
        elif self.fidelity_type == 'multi_fidelity':
            pass

        end_time = time.time()
        runtime = end_time - start_time

        # Record runtime using the Logger class
        self.logger.store_runtime_to_csv(runtime, self.log_path)
        self.pbar.close()


    def bestconfig_search_config(self, budget, fidelity, init_configs=None, kd_corr=1, fidelity_id=0, evaluated_filtered_configs=None):
        """
        :param budget:
        :param fidelity:
        :param init_configs:
        :param kd_corr:
        :param fidelity_id:
        :param lf_filtered_configs:
        :return:
        """

        consumed_iters = 0
        best_config = None
        best_perf = -math.inf if self._is_maximize_objective() else math.inf

        # Initial DDS sampling
        sampled_configs = self.dds_sampling(self.sampling_size)
        evaluated_configs, init_eval_count = self.evaluate_configs(sampled_configs, fidelity)
        consumed_iters += init_eval_count

        # if there exist configs that have been evaluated by low fidelity, evaluated it under current fidelity and
        # combine it with initial configs
        if evaluated_filtered_configs:

            # Combine and sort the results from both evaluations
            evaluated_configs = evaluated_configs + evaluated_filtered_configs
            if self._is_maximize_objective():
                evaluated_configs.sort(key=lambda x: x[1], reverse=True)
            else:
                evaluated_configs.sort(key=lambda x: x[1])

        evaluated_configs = evaluated_configs[:self.sampling_size]

        # initialize the best config
        for config, perf, _ in evaluated_configs:
            if self._is_better(perf, best_perf):
                best_perf = perf
                best_config = config

        while consumed_iters < budget:
            found_better = False
            for round_counter in range(self.max_rounds):
                # RBS is used to determine the boundary
                bounded_space = self.define_bounded_space(best_config, evaluated_configs)

                # sampling configs under the boundary
                bounded_sampled_configs = self.dds_sampling_within_bounds(bounded_space, self.sampling_size)
                new_evaluated_configs, new_eval_count = self.evaluate_configs(bounded_sampled_configs, fidelity)
                consumed_iters += new_eval_count

                # update the best config
                for config, perf, _ in new_evaluated_configs:
                    if self._is_better(perf, best_perf):
                        best_perf = perf
                        best_config = config
                        found_better = True

                if consumed_iters >= budget:
                    return best_config, best_perf

                if not found_better:
                    break

            # if budget are not exhausted, resampling some configs from scratch
            sampled_configs = self.dds_sampling(self.sampling_size)
            evaluated_configs, global_eval_count = self.evaluate_configs(sampled_configs, fidelity)
            consumed_iters += global_eval_count

            for config, perf, _ in evaluated_configs:
                if self._is_better(perf, best_perf):
                    best_perf = perf
                    best_config = config

        return best_config, best_perf

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
            'throughput',
            'qps',
            'rps',
            'encodefps',
        }

    def _is_better(self, candidate, current_best):
        if self._is_maximize_objective():
            return candidate > current_best
        return candidate < current_best

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

    def sampling_configs(self, sampling_size):
        """sampling configs randomly,make ensure without repeat configs"""
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
                # TODO: extend other types of configs
            config_tuple = tuple(config.items())
            if config_tuple not in seen_configs:
                init_configs.append(config)
                seen_configs.add(config_tuple)

        return init_configs

    def dds_sampling(self, sampling_size):
        """
        Use Divide & Diverge Sampling (DDS) to generate configs
        each dimension is divided into sampling_size area and chose a value within each area
        """
        # obtain the partitions of each parameter
        knobs_info = self.target_system.knobs_info
        partitions = {knob_name: self.get_intervals(knob_info, sampling_size)
                      for knob_name, knob_info in knobs_info.items()}

        # initialize selectable indices of each dimension
        available_indices = {knob_name: list(range(sampling_size)) for knob_name in partitions.keys()}

        samples = []
        for _ in range(sampling_size):
            config = {}
            for knob_name, intervals in partitions.items():

                index = random.choice(available_indices[knob_name])
                lower, upper = intervals[index]
                knob_type = str(knobs_info[knob_name].get('type', '')).lower()
                if knob_type == 'integer':
                    value = round(random.uniform(lower, upper))
                elif knob_type == 'float':
                    value = random.uniform(lower, upper)
                elif knob_type in ('enum', 'boolean'):
                    value = self.map_to_enum(index, self._categorical_values(knobs_info[knob_name]), sampling_size)
                else:
                    raise ValueError(f"Unsupported knob type: {knobs_info[knob_name]['type']}")
                config[knob_name] = value

                available_indices[knob_name].remove(index)
            samples.append(config)

        return samples

    @staticmethod
    def get_intervals(knob_info, sampling_size):
        """
        divide parameter range into different area according to knob's type
        """
        knob_type = str(knob_info.get('type', '')).lower()
        if knob_type in ['integer', 'float']:
            step = (knob_info['max'] - knob_info['min']) / sampling_size
            return [(knob_info['min'] + step * i, knob_info['min'] + step * (i + 1)) for i in range(sampling_size)]
        elif knob_type in ('enum', 'boolean'):
            enum_values = build_categorical_values(knob_info)
            intervals = []
            for i in range(sampling_size):
                value_index = BestConfigTuner.map_sample_index_to_bucket(
                    i, len(enum_values), sampling_size
                )
                intervals.append((value_index, value_index))
            return intervals
        else:
            raise ValueError(f"Unsupported knob type: {knob_info['type']}")


    @staticmethod
    def map_sample_index_to_bucket(index, num_values, sampling_size):
        if num_values <= 0:
            raise ValueError("Categorical knob must provide at least one value.")
        if sampling_size <= 0:
            raise ValueError("sampling_size must be positive.")
        return min((index * num_values) // sampling_size, num_values - 1)


    @staticmethod
    def map_to_enum(index, enum_values, sampling_size):
        """
        mapping sample area into concrete enum value
        """
        enum_index = BestConfigTuner.map_sample_index_to_bucket(
            index, len(enum_values), sampling_size
        )
        return enum_values[enum_index]


    def dds_sampling_within_bounds(self, bounded_space, sampling_size):
        """
        use DDS to sample configs within boundary
        """
        knobs_info = self.target_system.knobs_info
        partitions = {knob_name: self.get_partitions_within_bounds(knob_name, bounds, sampling_size)
                      for knob_name, bounds in bounded_space.items()}

        # initialize selectable indices of each dimension
        available_indices = {knob_name: list(range(sampling_size)) for knob_name in partitions.keys()}
        samples = []

        for _ in range(sampling_size):
            config = {}
            for knob_name, intervals in partitions.items():
                index = random.choice(available_indices[knob_name])
                lower, upper = intervals[index]

                knob_type = str(knobs_info[knob_name].get('type', '')).lower()
                if knob_type == 'integer':
                    value = round(random.uniform(lower, upper))
                elif knob_type == 'float':
                    value = random.uniform(lower, upper)
                elif knob_type in ('enum', 'boolean'):
                    value = self.map_to_enum(index, self._categorical_values(knobs_info[knob_name]), sampling_size)
                else:
                    raise ValueError(f"Unsupported knob type: {knobs_info[knob_name]['type']}")
                config[knob_name] = value
                available_indices[knob_name].remove(index)
            samples.append(config)

        return samples

    def get_partitions_within_bounds(self, knob_name, bounds, sampling_size):
        """
        :param bounds: (lower_bound, upper_bound)
        :param knob_name
        :param knob_info:
        :param sampling_size:
        :return:
        """
        knob_info = self.target_system.knobs_info[knob_name]
        lower_bound, upper_bound = bounds

        knob_type = str(knob_info.get('type', '')).lower()
        if knob_type in ['integer', 'float']:
            step = (upper_bound - lower_bound) / sampling_size
            return [(lower_bound + step * i, lower_bound + step * (i + 1)) for i in range(sampling_size)]

        elif knob_type in ('enum', 'boolean'):
            intervals = []
            for i in range(sampling_size):
                value_index = self.map_sample_index_to_bucket(
                    i, len(self._categorical_values(knob_info)), sampling_size
                )
                intervals.append((value_index, value_index))
            return intervals
        else:
            raise ValueError(f"Unsupported knob type: {knob_info['type']}")


    def define_bounded_space(self, best_config, evaluated_configs):
        """
        Define RBS boundary according to current best config and historical measurements.
        :param best_config: current best config
        :param evaluated_configs: historical measurements [(config, perf, cost), ...]
        :return: boundary of each dimension {knob_name: (lower_bound, upper_bound)}
        """
        bounded_space = {}

        for knob_name, best_value in best_config.items():
            knob_info = self.target_system.knobs_info.get(knob_name, {})
            knob_type = knob_info.get("type", "float")

            if knob_type in ("enum", "boolean"):
                enum_values = self._categorical_values(knob_info)
                if best_value not in enum_values:
                    # fallback for safety
                    bounded_space[knob_name] = (best_value, best_value)
                    continue

                idx = enum_values.index(best_value)
                lower_bound = enum_values[idx - 1] if idx > 0 else best_value
                upper_bound = enum_values[idx + 1] if idx < len(enum_values) - 1 else best_value

            else:
                lower_bound = max([config[knob_name] for config, _, _ in evaluated_configs if config[knob_name] < best_value], default=best_value)
                upper_bound = min([config[knob_name] for config, _, _ in evaluated_configs if config[knob_name] > best_value], default=best_value)

            bounded_space[knob_name] = (lower_bound, upper_bound)

        return bounded_space

    @staticmethod
    def _categorical_values(knob_info):
        return build_categorical_values(knob_info)
