# -*- coding: utf-8 -*-

import configparser
import json
import os
from collections import defaultdict

from utils.path_layout import repo_root, resolve_repo_path, resolve_repo_script


class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


knob_config = {}

default_value = {'lua_path': 'oltp_read_write'}

auto_setting = ['knob_num', 'initial_tunable_knob_num']

REPO_PATH_KEYS = {
    "database": ["knob_config_file", "backup_path", "temp_config_file_path", "machine_profile"],
    "workload": ["fidelity_factor_file"],
    "tune": ["dependency_rule_file"],
}

SCRIPT_PATH_KEYS = {
    "workload": ["lua_path"],
}

ENV_OVERRIDES = {
    "database": {
        "host": "CT_DB_HOST",
        "port": "CT_DB_PORT",
        "user": "CT_DB_USER",
        "password": "CT_DB_PASSWORD",
        "sudopassword": "CT_DB_SUDO_PASSWORD",
        "container_name": "CT_DB_CONTAINER_NAME",
        "url": "CT_DB_URL",
        "docker_image": "CT_DB_DOCKER_IMAGE",
        "docker_memory": "CT_DB_DOCKER_MEMORY",
        "docker_cpus": "CT_DB_DOCKER_CPUS",
        "ready_timeout": "CT_DB_READY_TIMEOUT",
    }
}


def get_default_dict(dic):
    config_dic = defaultdict(str)
    for k in dic:
        config_dic[k] = dic[k]

    for key in default_value.keys():
        if key not in config_dic.keys() or config_dic[key] == '':
            config_dic[key] = default_value[key]
    return config_dic


def _merge_section(target, source):
    if not source:
        return
    for key, value in source.items():
        target[key] = value


def _resolve_paths(config_dict, config_dir):
    for section, keys in REPO_PATH_KEYS.items():
        values = config_dict.get(section, {})
        for key in keys:
            raw = values.get(key, "")
            if raw:
                values[key] = resolve_repo_path(raw, base_dir=config_dir)

    for section, keys in SCRIPT_PATH_KEYS.items():
        values = config_dict.get(section, {})
        for key in keys:
            raw = values.get(key, "")
            if raw:
                values[key] = resolve_repo_script(raw, base_dir=config_dir)


def _apply_env_overrides(config_dict):
    import os

    for section, mapping in ENV_OVERRIDES.items():
        values = config_dict.get(section, {})
        for key, env_name in mapping.items():
            env_value = os.getenv(env_name)
            if env_value is not None and env_value != "":
                values[key] = env_value


def parse_args(file):
    config_path = resolve_repo_path(file, prefer_existing=True)
    cf = DictParser()
    cf.read(config_path, encoding="utf-8")
    config_dict = cf.read_dict()

    machine_profile = config_dict.get("database", {}).get("machine_profile", "").strip()
    if machine_profile:
        machine_path = resolve_repo_path(machine_profile, base_dir=os.path.dirname(config_path))
        machine_cf = DictParser()
        machine_cf.read(machine_path, encoding="utf-8")
        machine_dict = machine_cf.read_dict()
        for section in ("database", "workload", "tune"):
            _merge_section(config_dict.setdefault(section, {}), machine_dict.get(section, {}))

    _apply_env_overrides(config_dict)
    _resolve_paths(config_dict, os.path.dirname(config_path))
    config_dict.setdefault("database", {})["config_file"] = config_path
    config_dict["database"]["repo_root"] = repo_root()

    global knob_config
    with open(config_dict['database']['knob_config_file'], "r", encoding="utf-8") as f:
        knob_config = json.load(f)

    return get_default_dict(config_dict["database"]), get_default_dict(config_dict['workload']), get_default_dict(config_dict['tune'])
