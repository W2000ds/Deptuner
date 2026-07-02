import os
import csv
import logging
from datetime import datetime


class Logger:
    def __init__(self, target_system, optimize_objective, workload_controller):
        self.target_system = target_system
        self.optimize_objective = optimize_objective
        self.workload_controller = workload_controller

    def _expected_header(self):
        return [
            self.optimize_objective,
            'latency',
            'throughput',
            'qps',
            'cost',
            'fidelity',
            'prepare_time',
            'run_time',
            'clean_time',
        ] + list(self.target_system.knobs_info.keys())

    def ensure_log_file_exists(self, file_path, file_name):
        """make sure the log file exist, otherwise, create it"""
        full_path = os.path.join(file_path, file_name)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        expected_header = self._expected_header()
        if os.path.exists(full_path):
            with open(full_path, 'r', newline='') as file:
                reader = csv.reader(file)
                existing_header = next(reader, [])
            if existing_header != expected_header:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{full_path}.legacy_{ts}"
                os.rename(full_path, backup_path)
                logging.warning(
                    "Detected incompatible CSV header in %s. Moved old file to %s",
                    full_path, backup_path
                )

        if not os.path.exists(full_path):
            with open(full_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(expected_header)

    def store_config_pop_to_csv(self, evaluated_config_pop, generation, fidelity, kd_corr, fidelity_id, log_path):
        """store configuration population into csv file, including generation info (for GA_Tuner)"""
        if fidelity_id == 0:
            file_path = os.path.join(log_path, f'hf_config_pop_gen_{generation}.csv')
        else:
            # Low fidelity settings are distinguished from each other by ids
            file_path = os.path.join(log_path, f'lf_id{fidelity_id}_config_pop_gen_{generation}.csv')

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            header = list(self.target_system.knobs_info.keys()) + [self.optimize_objective, 'evaluated_time', 'fidelity_setting', 'corr']
            writer.writerow(header)

            for config, perf, evaluated_time in evaluated_config_pop:
                factors = list(fidelity.values())
                row = [config[param] for param in self.target_system.knobs_info.keys()] + [perf, evaluated_time, factors, kd_corr]
                writer.writerow(row)

    def verified_config_pop_to_csv(self, evaluated_config_pop, fidelity, kd_corr, log_path, log_file):
        """store configuration population into csv file, including generation info (for GA_Tuner)"""

        file_path = os.path.join(log_path, log_file)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            header = list(self.target_system.knobs_info.keys()) + [self.optimize_objective, 'evaluated_time', 'fidelity_setting', 'corr']
            writer.writerow(header)

            for config, perf, evaluated_time in evaluated_config_pop:
                factors = list(fidelity.values())
                row = [config[param] for param in self.target_system.knobs_info.keys()] + [perf, evaluated_time, factors, kd_corr]
                writer.writerow(row)

    def logging_data(
        self,
        config,
        perf,
        evaluated_cost,
        fidelity,
        prepare_time,
        run_time,
        clean_time,
        log_path,
        log_file,
        latency=None,
        throughput=None,
        qps=None,
    ):
        # config is a dic -> knob_name : knob_value
        """record the config and performance into file"""
        self.ensure_log_file_exists(log_path, log_file)
        log_file = os.path.join(log_path, log_file)
        with open(log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            factors = list(fidelity.values())
            row = [
                perf,
                latency if latency is not None else "",
                throughput if throughput is not None else "",
                qps if qps is not None else "",
                evaluated_cost,
                factors,
                prepare_time,
                run_time,
                clean_time,
            ] + [config[param] for param in self.target_system.knobs_info.keys()]
            writer.writerow(row)

    @staticmethod
    def logging_dependency_state(row, log_path, log_file="dep_state.csv"):
        os.makedirs(log_path, exist_ok=True)
        file_path = os.path.join(log_path, log_file)
        file_exists = os.path.isfile(file_path)
        header = [
            "iter",
            "rule_id",
            "prior_exist",
            "prior_fail",
            "p_exist",
            "p_fail_exist",
            "p_bg",
            "risk_i",
            "state",
            "n_evidence",
            "violated",
            "hard_rejected",
            "credit",
            "outcome",
            "evidence_signal",
            "exist_update_reason",
            "p_exist_delta",
            "impact_update_reason",
            "fail_update_reason",
            "p_fail_delta",
        ]
        if file_exists:
            with open(file_path, "r", newline="") as file:
                reader = csv.reader(file)
                existing_header = next(reader, [])
            if existing_header != header:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{file_path}.legacy_{ts}"
                os.rename(file_path, backup_path)
                logging.warning(
                    "Detected incompatible dependency CSV header in %s. Moved old file to %s",
                    file_path,
                    backup_path,
                )
                file_exists = False
        with open(file_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def logging_cyber_twin2(self, config, perf, evaluated_cost, fidelity, prepare_time, run_time, clean_time, cyber_twin_path, cyber_twin_file):
        # config is a dic -> knob_name : knob_value
        """record the config and performance into file"""
        self.ensure_log_file_exists(cyber_twin_path, cyber_twin_file)
        cyber_twin_file = os.path.join(cyber_twin_path, cyber_twin_file)
        with open(cyber_twin_file, 'a', newline='') as file:
            writer = csv.writer(file)
            factors = list(fidelity.values())
            row = [config[param] for param in self.target_system.knobs_info.keys()] + [perf, evaluated_cost,
                                                                                       factors, prepare_time,
                                                                                       run_time, clean_time]
            writer.writerow(row)

    def logging_cyber_twin(self, config, perf, evaluated_cost, fidelity, prepare_time, run_time, clean_time,
                           cyber_twin_path, cyber_twin_file):
        """Record the config and performance into file only if it doesn't already exist."""
        self.ensure_log_file_exists(cyber_twin_path, cyber_twin_file)
        cyber_twin_file = os.path.join(cyber_twin_path, cyber_twin_file)

        # Load existing data and check for duplicates
        if os.path.exists(cyber_twin_file):
            with open(cyber_twin_file, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                num_knobs = len(self.target_system.knobs_info)
                fidelity_index = num_knobs + 2

                for row in reader:
                    if len(row) <= fidelity_index:
                        # Skip malformed/legacy rows with incompatible schema.
                        continue
                    stored_config = {header[i]: row[i] for i in range(num_knobs)}
                    try:
                        stored_fidelity = eval(row[fidelity_index])
                    except Exception:
                        continue

                    # Check if the current config and fidelity match the existing ones
                    if stored_config == config and stored_fidelity == list(fidelity.values()):
                        return  # If a match is found, skip logging to avoid duplicates

        # If no match is found, prepare to log the new entry
        factors = list(fidelity.values())
        new_row = [config[param] for param in self.target_system.knobs_info.keys()] + [
            perf, evaluated_cost, factors, prepare_time, run_time, clean_time
        ]

        # Write the new row to the file
        with open(cyber_twin_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

    def log_cost_related_factors(self, cost_related_factors, log_path, output_file='fidelity_factors_dva.csv'):
        """record the info for fidelity factors: cost related factors or not?"""
        full_path = os.path.join(log_path, output_file)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        all_factors = self.workload_controller.fidelity_factors_info.keys()

        with open(full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Factor Name', 'Is Cost Related'])
            for factor_name in all_factors:
                is_cost_related = 'Y' if factor_name in cost_related_factors else 'N'
                writer.writerow([factor_name, is_cost_related])

    @staticmethod
    def log_fidelity_ind_clusters(fidelity_inds, labels, log_path):
        """
        It's important for us to analysis if the clustering is reasonable?
        record the label of each fidelity individual
        :param log_path:
        :param fidelity_inds: fidelity individuals in the first front
        :param labels: the cluster label of each fidelity individual
        """
        file_path = os.path.join(log_path, f'fidelity_factors_dbscan.csv')
        #if not os.path.exists(log_path):
            #os.makedirs(log_path)

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            header = list(fidelity_inds[0][0].keys()) + ['corr', 'cost', 'cluster_label']
            writer.writerow(header)
            for individual, label in zip(fidelity_inds, labels):
                fidelity, corr, cost = individual
                row = [fidelity[param] for param in fidelity.keys()] + [corr, cost, label]
                writer.writerow(row)

    def store_fidelity_pop_to_csv(self, evaluated_fidelity_pop, pop_front_levels, generation, log_path, fidelity_metric):
        """
        Record information about fidelity population in each generation
        :param fidelity_metric: metric name for quantifying fidelity
        :param evaluated_fidelity_pop: population after NSGA-II selection
        :param pop_front_levels: corresponding front levels for each individual in the population
        :param generation: current generation number
        :param log_path: path to save the CSV file
        """
        file_path = os.path.join(log_path, f'fidelity_pop_gen_{generation}.csv')

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            header = list(self.workload_controller.fidelity_factors_info.keys()) + [fidelity_metric,
                                                                                    'evaluated_time', 'front_level']
            writer.writerow(header)
            for individual, front_level in zip(evaluated_fidelity_pop, pop_front_levels):
                fidelity, corr, evaluated_time = individual
                row = [fidelity[param] for param in self.workload_controller.fidelity_factors_info.keys()] + [corr,
                                                                                                              evaluated_time,
                                                                                                              front_level]
                writer.writerow(row)

    def retrieve_from_cyber_twin(self, config, fidelity, cyber_twin_path, cyber_twin_file):
        """Retrieve from Cyber-Twin.csv """
        cyber_twin_file = os.path.join(cyber_twin_path, cyber_twin_file)
        if not os.path.exists(cyber_twin_file):
            return None

        with open(cyber_twin_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)

            # obtain index
            num_knobs = len(self.target_system.knobs_info)
            perf_index = num_knobs
            evaluated_cost_index = num_knobs + 1
            fidelity_index = num_knobs + 2
            prepare_time_index = num_knobs + 3
            run_time_index = num_knobs + 4
            clean_time_index = num_knobs + 5

            for row in reader:
                stored_config = {header[i]: row[i] for i in range(num_knobs)}
                stored_fidelity = eval(row[fidelity_index])

                if stored_config == config and stored_fidelity == list(fidelity.values()):
                    perf = float(row[perf_index])
                    evaluated_cost = float(row[evaluated_cost_index])
                    prepare_time = float(row[prepare_time_index])
                    run_time = float(row[run_time_index])
                    clean_time = float(row[clean_time_index])
                    return (perf, evaluated_cost, prepare_time, run_time, clean_time)

        return None

    @staticmethod
    def store_runtime_to_csv(runtime, log_path):
        """
        Record the total runtime of the algorithm.
        :param runtime: Total runtime in seconds
        :param log_path: Path to save the CSV file
        """
        # Ensure the directory exists
        os.makedirs(log_path, exist_ok=True)
        file_path = os.path.join(log_path, 'runtime_record.csv')

        # Check if the file exists
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write the header only if the file doesn't exist
                writer.writerow(['runtime_seconds'])
            writer.writerow([runtime])
