import subprocess
import time
import shlex
import os

from utils.db_connector import DBConnector
from utils.db_connector import MysqlConnector
import re


class DBOperator:
    def __init__(self):
        self.db_connector = MysqlConnector()

    # TODO: need to specification
    def get_current_db_configurations(self):
        """get the current configs of mysql"""
        sql = "SHOW VARIABLES WHERE Variable_name IN （'innodb_buffer_pool_size', 'innodb_log_file_size', ...);"  # ...indicates other variable
        configurations = self.db_connector.execute(sql)
        return {config[0]: config[1] for config in configurations}

    def set_db_knob(self, knob_name, knob_value):
        """set the configs for mysql"""
        self.db_connector = MysqlConnector()
        self.db_connector.connect_db()
        sql = f"SET GLOBAL {knob_name} = {knob_value};"
        self.db_connector.execute(sql)
        self.db_connector.close_db()

    def evaluate_performance(self):
        """evaluate the performance of current configs"""
        avg_latency = 0
        transactions_per_sec = 0
        queries_per_sec = 0  # 初始化 queries_per_sec 变量
        # transfer the stream as line list
        lines = self.output.decode('utf-8').splitlines()

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
            elif 'queries:' in line and '(per sec.)' in line:
                queries_per_sec = float(line.split('(')[1].split(' ')[0])

        return avg_latency, transactions_per_sec, queries_per_sec

    def run_sysbench(self, factors):
        """
        Run sysbench benchmark test
        :param factors: fidelity factors (dic), which is using for adjust workload
        :return: latency, throughput, qps, prepare_time, run_time, clean_time
        """

        # fixed parameters for running worklaod
        host = '127.0.0.1'
        user = 'root'
        password = ''
        testdb = 'testdb'
        # threads = 10
        # mode = 'complex'

        cmd_prefix = f"sysbench oltp_read_write --mysql-host={host} --mysql-user={user} --mysql-password={password} --mysql-db={testdb} "

        for key, value in factors.items():
            cmd_prefix += f" --{key}={value}"

        cmd_cleanup = cmd_prefix + " cleanup"
        subprocess.call(cmd_cleanup, shell=True)

        # prepare data
        start_time = time.time()
        cmd_prepare = cmd_prefix + " prepare"
        subprocess.call(cmd_prepare, shell=True)
        end_time = time.time()
        prepare_time = end_time - start_time

        # run workload
        start_time = time.time()
        cmd_run = cmd_prefix + " run"
        # output = subprocess.getoutput(cmd_run)
        cmd_list = shlex.split(cmd_run)
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.output, self.error = process.communicate()
        end_time = time.time()
        run_time = end_time - start_time

        # get obj perf, e.g., throughput/latency/qps
        latency, throughput, qps = self.evaluate_performance()

        # clean data
        start_time = time.time()
        cmd_cleanup = cmd_prefix + " cleanup"
        subprocess.call(cmd_cleanup, shell=True)
        end_time = time.time()
        clean_time = end_time - start_time

        return float(latency), float(throughput), float(qps), float(prepare_time), float(run_time), float(clean_time)

    def get_db_knob_value(self, knob_name):
        conn = self.db_connector.connect()
        cur = conn.cursor()
        sql = "show variables like '{}';".format(knob_name)
        cur.execute(sql)
        result = cur.fetchall()
        cur.close()
        # conn.close()
        return result[0][1]

    # extent other methods
