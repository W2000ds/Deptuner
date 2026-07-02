import json
import subprocess
import time
import shlex
import os
import pandas as pd
import re
from utils.db_connector import DBConnector
from utils.db_connector import MysqlConnector
import xml.etree.ElementTree as ET
from utils.path_layout import resolve_repo_path, resolve_repo_script


class WorkloadController:
    def __init__(self, args_db, args_workload, target_system):
        self.error = None
        self.output = None
        self.workload_bench = args_workload['workload_bench']
        self.host = args_db['host']
        self.user = args_db.get('user', '')
        self.port = args_db.get('port', '')
        self.password = args_db.get('password', '')
        self.dbname = args_workload.get('dbname', '')
        self.sys_name = args_db['db']
        self.server_url = args_db.get('url', f"http://{self.host}:{self.port}/")
        self.lua_path = args_workload.get('lua_path', '')
        self.target_system = target_system

        # Fidelity Factors Info
        self.fidelity_factors_info = self.initialize_fidelity_factors(args_workload['fidelity_factor_file'],
                                                                      int(args_workload['fidelity_factor_num']))
        self.default_fidelity_factors = self.get_default_fidelity_factors()


    @staticmethod
    def initialize_fidelity_factors(fidelity_factor_file, fidelity_factor_num):
        """
        Initialize fidelity_factors, including the name, type, value and so on.
        """
        global FIDELITY_FACTORS
        global FIDELITY_FACTORS_INFO
        fidelity_factor_file = resolve_repo_path(fidelity_factor_file, prefer_existing=True)
        if fidelity_factor_num == -1:
            f = open(fidelity_factor_file)
            FIDELITY_FACTORS_INFO = json.load(f)
            FIDELITY_FACTORS = list(FIDELITY_FACTORS_INFO.keys())
            f.close()
        else:
            f = open(fidelity_factor_file)
            factor_tmp = json.load(f)
            i = 0
            FIDELITY_FACTORS_INFO = {}
            while i < fidelity_factor_num:
                key = list(factor_tmp.keys())[i]
                FIDELITY_FACTORS_INFO[key] = factor_tmp[key]
                i = i + 1
            FIDELITY_FACTORS = list(FIDELITY_FACTORS_INFO.keys())
            f.close()
        return FIDELITY_FACTORS_INFO


    @staticmethod
    def get_default_fidelity_factors():
        """
        Get default fidelity factors, which is considered as the highest fidelity, original task, i.e., true evaluation
        """
        default_factors = {}
        for name, value in FIDELITY_FACTORS_INFO.items():
            if not value['type'] == "combination":
                default_factors[name] = value['default']
            else:
                pass
        return default_factors

    def run_workload(self, factors):

        latency, throughput, qps, prepare_time, run_time, clean_time = 0, 0, 0, 0, 0, 0
        if self.workload_bench == 'sysbench':
            latency, throughput, qps, prepare_time, run_time, clean_time = self.run_sysbench(factors)
        elif self.workload_bench == 'tpcc':
            latency, throughput, prepare_time, run_time, clean_time = self.run_tpcc(factors)
            qps = 0  # TPCC doesn't have QPS metric, set to 0
        elif self.workload_bench == 'wrk':
            latency, throughput, qps, prepare_time, run_time, clean_time = self.run_wrk(factors)
        elif self.workload_bench == 'ab':
            latency, throughput, qps, prepare_time, run_time, clean_time = self.run_ab(factors)
        elif self.workload_bench == 'x265':
            latency, throughput, qps, prepare_time, run_time, clean_time = self.run_x265(factors)
        elif self.workload_bench == 'ycsb':
            pass

        return float(latency), float(throughput), float(qps), float(prepare_time), float(run_time), float(clean_time)

    def run_x265(self, factors):
        start = time.time()
        result = self.target_system.run_benchmark(factors)
        run_time = time.time() - start

        self.output = result.stdout.encode("utf-8", errors="ignore")
        self.error = result.stderr.encode("utf-8", errors="ignore")

        metrics = self.target_system.extract_data_from_output(result.stdout)
        encode_time = metrics["EncodeTime"]
        encode_fps = metrics["EncodeFPS"]

        print(
            "[x265] "
            f"EncodeFPS={metrics['EncodeFPS']:.2f}, "
            f"EncodeTime={metrics['EncodeTime']:.2f}s, "
            f"AvgCPUUtil={metrics['AvgCPUUtil']:.2f}%, "
            f"OutputFileMB={metrics['OutputFileMB']:.3f}, "
            f"BitrateKbps={metrics['BitrateKbps']:.2f}"
        )

        return encode_time, encode_fps, encode_fps, 0, run_time, 0

    def run_wrk(self, factors):
        command_parts = ["wrk"]
        if "threads" in factors:
            command_parts.extend(["-t", str(factors["threads"])])
        if "connections" in factors:
            command_parts.extend(["-c", str(factors["connections"])])
        if "duration" in factors:
            command_parts.extend(["-d", f"{factors['duration']}s"])
        if "timeout" in factors:
            command_parts.extend(["--timeout", str(factors["timeout"])])
        headers = factors.get("headers")
        if isinstance(headers, dict):
            for key, value in headers.items():
                command_parts.extend(["-H", f"{key}: {value}"])

        use_post = str(factors.get("post", "false")).lower() in ("true", "1", "yes", "on")
        wrk_script = str(factors.get("wrk_script", "")).strip()
        if wrk_script:
            command_parts.extend(["-s", resolve_repo_script(wrk_script)])
        elif use_post:
            command_parts.extend(["-s", "wrk/scripts/post.lua"])
        command_parts.append(self.server_url)
        cmd = " ".join(command_parts)

        start = time.time()
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.output, self.error = process.communicate()
        run_time = time.time() - start

        latency, rps = self.evaluate_performance_wrk(factors)
        return latency, rps, rps, 0, run_time, 0

    def evaluate_performance_wrk(self, factors=None):
        factors = factors or {}
        out = self.output.decode("utf-8", errors="ignore")
        rps_match = re.search(r"Requests/sec:\s+(\d+(\.\d+)?)", out)
        lat_match = re.search(r"Latency\s+(\d+(\.\d+)?)", out)
        rps = float(rps_match.group(1)) if rps_match else 0.0
        latency = float(lat_match.group(1)) if lat_match else 0.0
        fail_on_non_2xx = str(factors.get("fail_on_wrk_non_2xx", "false")).lower() in (
            "true",
            "1",
            "yes",
            "on",
        )
        non_2xx_match = re.search(r"Non-2xx or 3xx responses:\s+(\d+)", out)
        fail_on_socket_errors = str(factors.get("fail_on_wrk_socket_errors", "false")).lower() in (
            "true",
            "1",
            "yes",
            "on",
        )
        socket_error_match = re.search(r"Socket errors:\s+connect\s+(\d+),\s+read\s+(\d+),\s+write\s+(\d+),\s+timeout\s+(\d+)", out)
        has_non_2xx = bool(non_2xx_match and int(non_2xx_match.group(1)) > 0)
        has_socket_errors = False
        if socket_error_match:
            has_socket_errors = any(int(value) > 0 for value in socket_error_match.groups())
        if fail_on_non_2xx and has_non_2xx:
            print("[wrk] Non-2xx/3xx responses detected; returning zero throughput.")
            return latency, 0.0
        if fail_on_socket_errors and has_socket_errors:
            print("[wrk] Socket errors detected; returning zero throughput.")
            return latency, 0.0
        return latency, rps

    def run_ab(self, factors):
        requests = factors.get("requests", 100)
        concurrency = factors.get("concurrency", 10)
        timelimit = factors.get("timelimit", 30)
        use_post = str(factors.get("post", "false")).lower() in ("true", "1", "yes", "on")

        if use_post:
            cmd = (
                f'ab -n {requests} -c {concurrency} -t {timelimit} '
                f'-p post_data.txt -T "application/x-www-form-urlencoded" {self.server_url}'
            )
        else:
            cmd = f"ab -n {requests} -c {concurrency} -t {timelimit} {self.server_url}"

        start = time.time()
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.output = result.stdout.encode("utf-8")
        run_time = time.time() - start
        latency, rps = self.evaluate_performance_ab()
        return latency, rps, rps, 0, run_time, 0

    def evaluate_performance_ab(self):
        out = self.output.decode("utf-8", errors="ignore")
        rps_match = re.search(r"Requests per second:\s+(\d+(\.\d+)?)", out)
        tpr_match = re.search(r"Time per request:\s+(\d+(\.\d+)?)", out)
        rps = float(rps_match.group(1)) if rps_match else 0.0
        latency = float(tpr_match.group(1)) if tpr_match else 0.0
        return latency, rps

    def run_sysbench(self, factors):
        """
        Run customized sysbench workload (e.g., workload_table_cache_stress.lua)
        :param factors: fidelity factors (dict), e.g., {'tables': 16, 'table-size': 100000, 'threads': 128, 'time': 30}
        :return: latency, throughput, qps, prepare_time, run_time, clean_time
        """
        # Fast-fail guard to avoid sysbench thread-init storms when DB is unreachable.
        if hasattr(self.target_system, "check_connection_alive"):
            try:
                if not self.target_system.check_connection_alive():
                    print("[Sysbench] DB unreachable; skip sysbench run and return zero metrics.")
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            except Exception as e:
                print(f"[Sysbench] DB connectivity check failed: {e}")
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Detect system type
        if self.sys_name == 'mysql':
            cmd_prefix = (
                f"sysbench {self.lua_path} "
                f"--mysql-host={self.host} "
                f"--mysql-port={self.port} "
                f"--mysql-user={self.user} "
                f"--mysql-password={self.password} "
                f"--mysql-db={self.dbname} "
            )
        elif self.sys_name == 'postgresql':
            cmd_prefix = (
                f"sysbench {self.lua_path} "
                f"--db-driver=pgsql "
                f"--pgsql-host={self.host} "
                f"--pgsql-port={self.port} "
                f"--pgsql-user={self.user} "
                f"--pgsql-password={self.password} "
                f"--pgsql-db={self.dbname} "
                f"--auto-inc=true "
            )
        else:
            raise ValueError("Unsupported system type")

        # Dynamically add workload parameters
        for key, value in factors.items():
            if key == "r_ratio":  # read/write ratio handled by Lua or mapping
                if value == 0.5:
                    cmd_prefix += f" --point-selects=0"
                elif value == 0.6:
                    cmd_prefix += f" --point-selects=2"
                elif value == 0.7:
                    cmd_prefix += f" --point-selects=5"
                elif value == 0.8:
                    cmd_prefix += f" --point-selects=12"
                elif value == 0.9:
                    cmd_prefix += f" --point-selects=32"
            else:
                cmd_prefix += f" --{key}={value}"

        # 强制设置自定义 workload 的额外参数（确保 Lua 可识别）
        cmd_prefix += " --rand-type=uniform"

        # Skip drop/create cycle by default to avoid sudo-interactive paths.
        # sysbench prepare still refreshes benchmark tables for current factors.

        # ========== PREPARE PHASE ==========
        cleanup_before_prepare = cmd_prefix + " cleanup"
        subprocess.run(
            shlex.split(cleanup_before_prepare),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        start_time = time.time()
        cmd_prepare = cmd_prefix + " prepare"
        print(f"[Sysbench] Preparing dataset with: {cmd_prepare}")
        subprocess.call(cmd_prepare, shell=True)
        prepare_time = time.time() - start_time
        print(f"[Sysbench] Prepare done in {prepare_time:.2f} s")

        # ========== RUN PHASE ==========
        cmd_run = cmd_prefix + " run"
        print(f"[Sysbench] Running workload with: {cmd_run}")
        start_time = time.time()
        cmd_list = shlex.split(cmd_run)
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.output, self.error = process.communicate()
        run_time = time.time() - start_time

        # ========== PERFORMANCE METRICS ==========
        latency, throughput, qps = self.evaluate_performance_sysbench()
        print(f"[Sysbench] Run done in {run_time:.2f} s")
        print(f"[Sysbench] QPS={qps:.2f}, Throughput={throughput:.2f}, Latency={latency:.2f}")

        # For sysbench, we don't have a separate cleanup phase, so clean_time is 0
        clean_time = 0

        return latency, throughput, qps, prepare_time, run_time, clean_time

    def evaluate_performance_sysbench(self):
        """evaluate the performance of current configs"""
        avg_latency = 0
        transactions_per_sec = 0
        queries_per_sec = 0  # 初始化 queries_per_sec 变量

        # transfer the stream as line list
        lines = self.output.decode('utf-8').splitlines()

        print("==== SYSBENCH OUTPUT ====")
        print(self.output.decode("utf-8"))
        print("=========================")

        for line in lines:
            if 'min:' in line:
                min_latency = float(line.split(':')[1].strip())
            elif 'avg:' in line:
                avg_latency = float(line.split(':')[1].strip())
            elif 'max:' in line:
                max_latency = float(line.split(':')[1].strip())
            elif '95th percentile:' in line:
                percentile_95 = float(line.split(':')[1].strip())
            elif 'total time:' in line:
                total_time = float(line.split(':')[1].strip()[:-1])  # Remove 's'
            elif 'total number of events:' in line:
                total_events = int(line.split(':')[1].strip())
            elif 'transactions:' in line:
                transactions_per_sec = float(line.split('(')[1].split(' ')[0])
            elif 'queries:' in line and 'per sec.)' in line:
                queries_per_sec = float(line.split('(')[1].split(' ')[0])
            elif "read:" in line:
                read_ops = int(re.search(r'read:\s+(\d+)', line).group(1))
            elif "write:" in line:
                write_ops = int(re.search(r'write:\s+(\d+)', line).group(1))

        return avg_latency, transactions_per_sec, queries_per_sec

    def set_fidelity_factors(self, config_file, factors):
        """
        update the parameter of OLTPBench file (e.g., scale factor, time)
        :param config_file: file path
        :param factors: dic with fidelity-key and fidelity-value (e.g., scalefactor, time)
        """
        tree = ET.parse(config_file)
        root = tree.getroot()

        # set the scale factor and time dynamically (control the fidelity)
        for key, value in factors.items():
            for elem in root.iter(key):
                elem.text = str(value)

        # update the connection info of db
        for elem in root.iter('DBUrl'):
            elem.text = f"jdbc:{self.sys_name}://{self.host}:{self.port}/{self.dbname}"
        for elem in root.iter('username'):
            elem.text = self.user
        for elem in root.iter('password'):
            elem.text = self.password

        # save the updated config file
        tree.write(config_file)
        print(f"Updated {config_file} with factors: {factors}")

    
    @staticmethod
    def evaluate_performance_tpcc():

        res_file = "results/tpcc_result.res"  # file name is like tpcc_result.res
        csv_file = "results/tpcc_result.csv"

        if not os.path.exists(res_file):
            raise FileNotFoundError(f"{res_file} does not exist.")

        df = pd.read_csv(res_file)
        df.columns = df.columns.str.strip()  # remove the space in front of the column
        # print("Columns after cleaning:", list(df.columns))

        # calculate ave throughput and 95th avg latency (avg_lat)
        avg_throughput = df['throughput(req/sec)'].mean() if 'throughput(req/sec)' in df.columns else None
        avg_95th_latency = df["95th_lat(ms)"].mean()

        # delete .res file
        os.remove(res_file)
        os.remove(csv_file)

        return avg_95th_latency, avg_throughput
