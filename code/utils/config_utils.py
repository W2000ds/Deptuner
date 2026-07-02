import random

from pyDOE import lhs


class ConfigUtils:

    @staticmethod
    def get_top_k_configs(evaluated_configs, k=1, optimize_objective='throughput'):
        """
        Get top-k configurations based on the specified optimization objective.

        :param evaluated_configs: List of tuples (config, perf, cost)
        :param k: Number of top configurations to return
        :param optimize_objective: Objective to optimize ('throughput', 'RPS' for maximize; others for minimize)
        :return: List of top-k config dictionaries
        """
        if optimize_objective in ['throughput', 'RPS']:
            sorted_configs = sorted(evaluated_configs, key=lambda x: x[1], reverse=True)
        else:
            sorted_configs = sorted(evaluated_configs, key=lambda x: x[1])
        return [config for config, _, _ in sorted_configs[:k]]


    @staticmethod
    def sampling_configs_by_lhs(sample_size, knobs_info):
        """
        Get sample_size configurations based on latin hypercube sampling (lhs)

        :param sample_size:
        :param knobs_info:
        :return:
        """
        num_params = len(knobs_info)
        lhs_sample = lhs(num_params, samples=sample_size)
        configs = []

        for i in range(sample_size):
            config = {}
            for j, (key, val) in enumerate(knobs_info.items()):
                if val['type'] == 'integer':
                    range_width = val['max'] - val['min'] + 1
                    scaled_value = int(lhs_sample[i][j] * range_width) + val['min']
                    config[key] = scaled_value
                elif val['type'] == 'float':
                    range_width = val['max'] - val['min']
                    scaled_value = lhs_sample[i][j] * range_width + val['min']
                    config[key] = scaled_value
                elif val['type'] == 'enum':
                    possible_values = val['enum_values']
                    index = min(int(lhs_sample[i][j] * len(possible_values)), len(possible_values) - 1)
                    config[key] = possible_values[index]
            configs.append(config)

        return configs

    @staticmethod
    def sampling_configs_by_rs(sampling_size, knobs_info):
        """
        Get sample_size configurations based on random search (rs)

        :param sampling_size:
        :param knobs_info:
        :return:
        """
        init_configs = []
        seen_configs = set()
        while len(init_configs) < sampling_size:
            config = {}
            for knob_name, knob_info in knobs_info.items():
                if knob_info['type'] == 'integer':
                    config[knob_name] = random.randint(knob_info['min'], knob_info['max'])
                elif knob_info['type'] == 'float':
                    config[knob_name] = random.uniform(knob_info['min'], knob_info['max'])
                elif knob_info['type'] == 'enum':
                    config[knob_name] = random.choice(knob_info['enum_values'])
            config_tuple = tuple(config.items())
            if config_tuple not in seen_configs:
                init_configs.append(config)
                seen_configs.add(config_tuple)
        return init_configs


    @staticmethod
    def preprocess_configs_with_knobs_info(configs, knobs_info):
        """
        Preprocess config list into a numerical 2D list based on knobs_info.

        :param configs: List of config dicts: {config_name: config_value}
        :param knobs_info: Dict with info about each knob: {config_name: {"type": str, "enum_values": list (if enum)}}
        :return: List of processed configs in 2D list format
        """
        processed_configs = []
        for config in configs:
            processed_config = []
            for key, value in config.items():
                if key in knobs_info and knobs_info[key]["type"] == "enum":
                    enum_values = knobs_info[key]["enum_values"]
                    processed_config.append(enum_values.index(value))
                else:
                    processed_config.append(value)
            processed_configs.append(processed_config)
        return processed_configs


