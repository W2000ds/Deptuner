#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

from pyDOE import lhs

from utils.params_parsing import parse_args
from systems.factory import create_target_system
from utils.knob_space_utils import build_categorical_values


class ConfigSampler:
    def __init__(self, args_db, args_workload, args_tune, seed: int = 0):
        self.args_db = args_db
        self.args_workload = args_workload
        self.args_tune = args_tune
        self.target_system = create_target_system(args_db)
        self.rng = random.Random(seed)

    def sample_random(self, sample_size: int) -> list[dict]:
        samples = []
        seen = set()
        while len(samples) < sample_size:
            config = {}
            for knob_name, knob_info in self.target_system.knobs_info.items():
                knob_type = str(knob_info.get("type", "")).lower()
                if knob_type == "integer":
                    config[knob_name] = self.rng.randint(knob_info["min"], knob_info["max"])
                elif knob_type == "float":
                    config[knob_name] = self.rng.uniform(knob_info["min"], knob_info["max"])
                elif knob_type in ("enum", "boolean"):
                    values = build_categorical_values(knob_info)
                    config[knob_name] = self.rng.choice(values)
            key = tuple(sorted(config.items()))
            if key not in seen:
                seen.add(key)
                samples.append(config)
        return samples

    def sample_lhs(self, sample_size: int) -> list[dict]:
        num_params = len(self.target_system.knobs_info)
        lhs_sample = lhs(num_params, samples=sample_size)
        samples = []
        for row_idx in range(sample_size):
            config = {}
            for col_idx, (key, info) in enumerate(self.target_system.knobs_info.items()):
                knob_type = str(info.get("type", "")).lower()
                if knob_type == "integer":
                    width = info["max"] - info["min"] + 1
                    config[key] = int(lhs_sample[row_idx][col_idx] * width) + info["min"]
                elif knob_type == "float":
                    width = info["max"] - info["min"]
                    config[key] = lhs_sample[row_idx][col_idx] * width + info["min"]
                elif knob_type in ("enum", "boolean"):
                    values = build_categorical_values(info)
                    config[key] = values[min(int(lhs_sample[row_idx][col_idx] * len(values)), len(values) - 1)]
            samples.append(config)
        return samples

    def dds_sample(self, sample_size: int) -> list[dict]:
        knobs_info = self.target_system.knobs_info
        partitions = {
            knob_name: self._get_intervals(knob_info, sample_size)
            for knob_name, knob_info in knobs_info.items()
        }
        available = {knob_name: list(range(sample_size)) for knob_name in partitions}
        samples = []
        for _ in range(sample_size):
            config = {}
            for knob_name, intervals in partitions.items():
                index = self.rng.choice(available[knob_name])
                lower, upper = intervals[index]
                knob_type = str(knobs_info[knob_name].get("type", "")).lower()
                if knob_type == "integer":
                    value = round(self.rng.uniform(lower, upper))
                elif knob_type == "float":
                    value = self.rng.uniform(lower, upper)
                elif knob_type in ("enum", "boolean"):
                    value = self._map_to_enum(index, build_categorical_values(knobs_info[knob_name]), sample_size)
                else:
                    raise ValueError(f"Unsupported knob type: {knob_type}")
                config[knob_name] = value
                available[knob_name].remove(index)
            samples.append(config)
        return samples

    def _get_intervals(self, knob_info: dict, sample_size: int):
        knob_type = str(knob_info.get("type", "")).lower()
        if knob_type in {"integer", "float"}:
            step = (knob_info["max"] - knob_info["min"]) / sample_size
            return [(knob_info["min"] + step * i, knob_info["min"] + step * (i + 1)) for i in range(sample_size)]
        if knob_type in ("enum", "boolean"):
            values = build_categorical_values(knob_info)
            return [(self._map_bucket(i, len(values), sample_size), self._map_bucket(i, len(values), sample_size)) for i in range(sample_size)]
        raise ValueError(f"Unsupported knob type: {knob_info['type']}")

    @staticmethod
    def _map_bucket(index: int, num_values: int, sample_size: int) -> int:
        return min((index * num_values) // sample_size, num_values - 1)

    @classmethod
    def _map_to_enum(cls, index: int, enum_values: list, sample_size: int):
        return enum_values[cls._map_bucket(index, len(enum_values), sample_size)]


def load_sampler(config_path: str, seed: int = 0) -> ConfigSampler:
    args_db, args_workload, args_tune = parse_args(config_path)
    return ConfigSampler(args_db, args_workload, args_tune, seed=seed)


__all__ = ["ConfigSampler", "load_sampler"]
