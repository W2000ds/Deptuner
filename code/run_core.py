#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tuner.bestconfig_tuner import BestConfigTuner
from tuner.bo_tuner import BOTuner
from tuner.flash_tuner import FLASHTuner
from tuner.hebo_tuner import HEBOTuner
from tuner.promise_tuner import PromiseTuner
from tuner.tpe_tuner import TPETuner
from utils.params_parsing import parse_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="最小运行入口，仅保留核心调优器。")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--db_host", type=str, required=False, help="数据库主机")
    parser.add_argument("--fidelity_type", type=str, required=True, help="保真度类型")
    parser.add_argument("--tuning_method", type=str, required=True, help="调优方法")
    parser.add_argument("--run", type=str, required=True, help="运行编号")
    return parser


def normalize_host(args_db: dict[str, str]) -> None:
    if str(args_db.get("host", "")).strip() == "0.0.0.0":
        args_db["host"] = "127.0.0.1"


def create_optimizer(method: str, args_db, args_workload, args_tune, run: int):
    if method == "hebo":
        return HEBOTuner(args_db, args_workload, args_tune, run), "tune_hebo"
    if method == "bo":
        return BOTuner(args_db, args_workload, args_tune, run), "tune_bo"
    if method == "tpe":
        return TPETuner(args_db, args_workload, args_tune, run), "tune_tpe"
    if method == "flash":
        return FLASHTuner(args_db, args_workload, args_tune, run), "tune_flash"
    if method == "bestconfig":
        return BestConfigTuner(args_db, args_workload, args_tune, run), "tune_bestconfig"
    if method in {"promisetune", "promise"}:
        return PromiseTuner(args_db, args_workload, args_tune, run), "tune_promise"
    raise ValueError(f"Unsupported tuning method: {method}")


def main() -> None:
    opt = build_parser().parse_args()
    args_db, args_workload, args_tune = parse_args(opt.config)

    if opt.db_host is not None:
        args_db["host"] = opt.db_host
    normalize_host(args_db)

    args_tune["tuning_method"] = opt.tuning_method
    args_tune["fidelity_type"] = opt.fidelity_type
    run = int(opt.run)

    optimizer, method_name = create_optimizer(opt.tuning_method, args_db, args_workload, args_tune, run)
    getattr(optimizer, method_name)()


if __name__ == "__main__":
    main()
