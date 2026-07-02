import argparse
from tuner.hebo_tuner import HEBOTuner
from tuner.promise_tuner import PromiseTuner
from tuner.ga_tuner import GATuner
from tuner.bestconfig_tuner import BestConfigTuner
from tuner.flash_tuner import FLASHTuner
from tuner.bayes_tuner import BayesTuner
from tuner.bo_tuner import BOTuner
from tuner.default_test import DefaultTester
from tuner.dep_evidence_test import DependencyEvidenceTester
from utils.params_parsing import parse_args
from tuner.dep_test import DepTester
from tuner.tpe_tuner import TPETuner
from tuner.splo_random_tuner import SPLORandomTuner
from tuner.bdd_splo_tuner import BDDSPLTuner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./params_setup/tuning/mysql_hebo_dep.ini', help='config file')
    parser.add_argument('--db_host', type=str, required=False, help='Database host to use for this run')
    parser.add_argument('--fidelity_type', type=str, required=False, help='Fidelity type to use')
    parser.add_argument('--tuning_method', type=str, required=False, help='Tuning method to use (GA, GA3, etc.)')
    parser.add_argument('--run', type=str, required=False, help='No. of runs')
    opt = parser.parse_args()

    # parse the mysql_params_setup.ini file
    args_db, args_workload, args_tune = parse_args(opt.config)

    # dynamically set the corresponding db service, tuning method, fidelity type, and no. of run.
    if opt.db_host is not None:
        args_db['host'] = opt.db_host
    # 0.0.0.0 is a bind/listen address, not a client destination.
    if str(args_db['host']).strip() == '0.0.0.0':
        args_db['host'] = '127.0.0.1'
    args_tune['tuning_method'] = opt.tuning_method
    args_tune['fidelity_type'] = opt.fidelity_type
    run = int(opt.run)

    if args_tune['tuning_method'] == 'RS':
        optimizer = RandomSearchOptimizer(args_db, args_workload, args_tune, run)
        optimizer.search_best_config()
    elif args_tune['tuning_method'] == 'ga':
        optimizer = GATuner(args_db, args_workload, args_tune, run)
        optimizer.tune_ga()
    elif args_tune['tuning_method'] == 'flash':
        optimizer = FLASHTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_flash()
    elif args_tune['tuning_method'] == 'bestconfig':
        optimizer = BestConfigTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_bestconfig()
    elif args_tune['tuning_method'] == 'smac':
        optimizer = SMACTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_smac()
    elif args_tune['tuning_method'] == 'hyperband':
        optimizer = HBTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_hyperband()
    elif args_tune['tuning_method'] == 'mf_analyser':
        analyser = MFAnalyser(args_db, args_workload, args_tune, run)
        analyser.analyse_and_verify_configs()
    elif args_tune['tuning_method'] == 'mf_sampler':
        collector = MFSampler(args_db, args_workload, args_tune, run)
        collector.sampling_and_evaluate()
    elif args_tune['tuning_method'] == 'mf_flash':
        optimizer = MFFLASHTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_mf_flash()
    elif args_tune['tuning_method'] == 'bayes':
        optimizer = BayesTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_bayes()
    elif args_tune['tuning_method'] == 'bo':
        optimizer = BOTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_bo()
    elif args_tune['tuning_method'] == 'tpe':
        optimizer = TPETuner(args_db, args_workload, args_tune, run)
        optimizer.tune_tpe()
    elif args_tune['tuning_method'] == 'srs':
        optimizer = SPLORandomTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_srs()
    elif args_tune['tuning_method'] == 'rrs':
        optimizer = SPLORandomTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_rrs()
    elif args_tune['tuning_method'] == 'bdd_srs':
        optimizer = BDDSPLTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_bdd_srs()
    elif args_tune['tuning_method'] == 'bdd_rrs':
        optimizer = BDDSPLTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_bdd_rrs()
    elif args_tune['tuning_method'] == 'bohb':
        optimizer = BOHBTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_bohb()
    elif args_tune['tuning_method'] == 'dehb':
        optimizer = DEHBTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_dehb()
    elif args_tune['tuning_method'] == 'hebo':
        optimizer = HEBOTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_hebo()
    elif args_tune['tuning_method'] in ('promisetune', 'promise'):
        optimizer = PromiseTuner(args_db, args_workload, args_tune, run)
        optimizer.tune_promise()
    elif args_tune['tuning_method'] == 'default':
        optimizer = DefaultTester(args_db, args_workload, args_tune, run)
        optimizer.run_default()
    elif args_tune['tuning_method'] == 'deptest':
        optimizer = DepTester(args_db, args_workload, args_tune, run)
        optimizer.run_default()
    elif args_tune['tuning_method'] == 'depevidence':
        optimizer = DependencyEvidenceTester(args_db, args_workload, args_tune, run)
        optimizer.run_default()


if __name__ == "__main__":
    main()
