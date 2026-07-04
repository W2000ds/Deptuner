# Deptune Supplementary Repository

This repository contains the supplementary code, experimental results, and supporting material for Deptune.

Deptune is a dependency-aware configuration tuning framework for software systems. The repository includes online tuning code and the per-measurement CSV results used for the paper's research questions.

## Repository Layout

```text
Deptune/
├── code/      # Core online tuning implementation
├── RQs/       # Processed CSV results organized by research question
├── Rawdata/   # Reserved for raw or additional source data
└── support/   # Supporting documents
```

## Code

The `code/` directory contains the runtime implementation copied from the main project.

Key components:

- `code/main.py`: command-line entry point for running tuners.
- `code/tuner/`: tuning algorithms, including HEBO, BO, TPE, BestConfig, PromiseTune, FLASH, and related samplers.
- `code/systems/`: system adapters for MySQL, PostgreSQL, HTTPD, Tomcat, and x265.
- `code/workload.py` and `code/workload/`: workload execution logic and benchmark support.
- `code/dependency/`: dependency-rule handling and runtime evidence extraction.
- `code/utils/`: parameter parsing, logging, knob normalization, and shared utilities.
- `code/lua/` and `code/wrk/`: workload scripts used by database and web-service experiments.

Example command:

```bash
cd code
python main.py \
  --config path/to/config.ini \
  --db_host 127.0.0.1 \
  --fidelity_type single_fidelity \
  --tuning_method hebo \
  --run 1
```

Install Python dependencies with:

```bash
pip install -r code/requirements.txt
```

Running online tuning also requires the corresponding target system, workload driver, containers or services, and configuration files.

## Results

The `RQs/` directory stores processed CSV files. Results are organized by research question, then by target system.

Target systems:

- `mysql`
- `postgresql`
- `httpd`
- `tomcat`
- `x265`

### RQ1

`RQs/RQ1/` contains HEBO, TPE, and simulated BO results.

Each system directory contains six files:

- `bo_dep.csv`
- `bo_plain.csv`
- `hebo_dep.csv`
- `hebo_plain.csv`
- `tpe_dep.csv`
- `tpe_plain.csv`

Each row corresponds to one measurement. The CSV files include metadata columns such as `system`, `method`, `run_id`, `variant`, and `iteration`, performance columns such as `qps`, `latency`, `throughput`, and `cost`, plus `config_json` for the full configuration choice.

### RQ2

`RQs/RQ2/` contains baseline and sampler-based method results.

Each system directory contains:

- `bestconfig.csv`
- `flash.csv`
- `promisetune.csv`
- `spl.csv`

The `spl.csv` files correspond to SRS+BDDSampler / SPL results.

### RQ3

`RQs/RQ3/` contains results for different low-fidelity and high-fidelity dependency settings.

Each system directory contains:

- `lf01_hard01.csv`
- `lf05_hard07.csv`
- `lf09_hard01.csv`
- `lf09_hard09.csv`

### RQ4

`RQs/RQ4/` contains ablation results.

Each system directory contains:

- `without_impact.csv`
- `without_reliability.csv`

## CSV Format

Most result CSV files follow this structure:

- `system`: target system.
- `method`: tuning method or experimental condition.
- `run_id`: run identifier.
- `variant`: configuration of the method, such as `depaware`, `plain`, or `single`.
- `iteration`: measurement index within a run.
- `qps`, `latency`, `throughput`, `cost`: performance metrics.
- `fidelity`, `prepare_time`, `run_time`, `clean_time`: workload execution details when available.
- `config_json`: complete configuration used for that measurement.

There is intentionally no `source_file` column in the released CSV files.

## Supporting Material

The `support/` directory contains additional paper-related material, including `Prompt1.pdf`.

## Notes

- The CSV files are already processed into paper-facing layouts and do not require raw log parsing for common analysis.
- The code directory is intended to show and run the core online tuning implementation.
- Environment-specific credentials, service addresses, and container settings should be supplied outside this repository.
