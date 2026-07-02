# Core Online Tuning Code

This folder contains the original project code needed for online tuning and configuration sampling.

## Included Modules

- `main.py`: command-line entry point for selecting and running a tuner.
- `tuner/`: online tuning algorithms, including HEBO, BO, TPE, BestConfig, PromiseTune, FLASH, and related samplers.
- `systems/`: adapters for MySQL, PostgreSQL, HTTPD, Tomcat, and x265.
- `workload.py` and `workload/`: workload execution controllers and benchmark support code.
- `dependency/`: dependency-aware rule handling and runtime evidence extraction.
- `utils/`: parameter parsing, logging, knob normalization, and shared runtime utilities.
- `lua/` and `wrk/`: workload scripts used by database and web-service experiments.

## Main Algorithms

- HEBO: `tuner/hebo_tuner.py`
- BO: `tuner/bo_tuner.py`
- TPE: `tuner/tpe_tuner.py`
- BestConfig: `tuner/bestconfig_tuner.py`
- PromiseTune: `tuner/promise_tuner.py`
- FLASH: `tuner/flash_tuner.py`

## Sampling Code

- Random sampling: `tuner/flash_tuner.py`, `tuner/promise_tuner.py`
- DDS sampling: `tuner/bestconfig_tuner.py`
- LHS sampling: `tuner/mf_sampler.py`
- Optimizer proposal sampling: `tuner/hebo_tuner.py`, `tuner/bo_tuner.py`, `tuner/tpe_tuner.py`

## Example

```bash
python main.py \
  --config path/to/config.ini \
  --db_host 127.0.0.1 \
  --fidelity_type single_fidelity \
  --tuning_method hebo \
  --run 1
```

Supported systems are MySQL, PostgreSQL, HTTPD, Tomcat, and x265.
