import json
import os
import re
import shlex
import subprocess
import time
from systems.base import BaseSystemAdapter
from utils.knob_space_utils import is_unset_value


class HttpdDB(BaseSystemAdapter):
    def __init__(self, args):
        self.args = args
        self.sys_name = "httpd"
        self.host = args.get("host", "127.0.0.1")
        self.port = args.get("port", "8080")
        self.user = args.get("user", "")
        self.password = args.get("password", "")
        self.sudopassword = args.get("sudopassword", "")
        self.server_url = args.get("url", f"http://{self.host}:{self.port}/")
        self.container_name = args.get("container_name", args.get("dockername", "httpd"))
        self.config_file_path = args.get("config_file_path", "/usr/local/apache2/conf/httpd.conf")
        self.backup_path = args.get("backup_path", "./tempfiles/httpd_backup.conf")
        self.temp_path = args.get("temp_config_file_path", "./tempfiles/httpd_temp.conf")
        self.pristine_backup_path = f"{self.backup_path}.pristine"
        self.ready_timeout = int(args.get("ready_timeout", 60))

        self.knobs_info = self.load_knobs(args["knob_config_file"], int(args["knob_num"]))
        self.default_knobs = self.build_default_knobs(self.knobs_info)
        self.current_config = dict(self.default_knobs)

    def _run(self, cmd, check=True):
        return subprocess.run(
            cmd,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def collect_runtime_evidence(self, since=None, tail=200):
        parts = []
        config_text = self._format_dependency_evidence()
        if config_text:
            parts.append({"source": "config_state", "text": config_text})
        since_clause = ""
        if since is not None:
            try:
                since_clause = f" --since {shlex.quote(str(int(float(since))))}"
            except Exception:
                since_clause = ""
        cmd = self._sudo(
            f"docker logs --tail {int(tail)}{since_clause} {shlex.quote(self.container_name)} 2>&1"
        )
        res = self._run(cmd, check=False)
        text = (res.stdout or "").strip()
        if text:
            parts.append({"source": "server_log", "text": text})
        if res.stderr and res.stderr.strip():
            parts.append({"source": "server_log_stderr", "text": res.stderr.strip()})
        return parts

    def _format_dependency_evidence(self):
        cfg = dict(getattr(self, "current_config", {}) or {})
        keepalive = str(cfg.get("KeepAlive", "")).strip().lower()
        mkar = cfg.get("MaxKeepAliveRequests", "")
        timeout = cfg.get("KeepAliveTimeout", "")
        lines = [
            "HTTPD_CONFIG_SNAPSHOT " + json.dumps(cfg, sort_keys=True)
        ]
        if keepalive == "on":
            lines.append(
                "HTTPD_CONFIG_EVIDENCE httpd_keepalive_mkar_coupling active: "
                f"KeepAlive=On enables persistent connections; MaxKeepAliveRequests={mkar} controls the request cap per kept-alive connection."
            )
            lines.append(
                "HTTPD_CONFIG_EVIDENCE httpd_keepalive_timeout_coupling active: "
                f"KeepAlive=On enables persistent connections; KeepAliveTimeout={timeout} controls idle connection lifetime."
            )
        else:
            lines.append(
                "HTTPD_CONFIG_EVIDENCE httpd_keepalive_mkar_coupling inactive: KeepAlive is not On; MaxKeepAliveRequests is not on the persistent-connection path."
            )
            lines.append(
                "HTTPD_CONFIG_EVIDENCE httpd_keepalive_timeout_coupling inactive: KeepAlive is not On; KeepAliveTimeout is not on the persistent-connection path."
            )
        return "\n".join(lines)

    def _sudo(self, cmd):
        if self.sudopassword:
            return f"echo {shlex.quote(str(self.sudopassword))} | sudo -S {cmd}"
        return cmd

    def _ensure_local_parent(self, path):
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)

    def _docker_inspect_running(self):
        cmd = self._sudo(f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(self.container_name)}")
        res = self._run(cmd, check=False)
        if res.returncode != 0:
            return False
        return res.stdout.strip().lower() == "true"

    def start_container(self):
        if self._docker_inspect_running():
            return
        cmd = self._sudo(f"docker start {shlex.quote(self.container_name)}")
        res = self._run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"启动 httpd 容器失败: {self.container_name}, stderr={res.stderr.strip()}"
            )
        self._wait_until_ready()

    def stop_container(self):
        if not self._docker_inspect_running():
            return
        cmd = self._sudo(f"docker stop {shlex.quote(self.container_name)}")
        res = self._run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"停止 httpd 容器失败: {self.container_name}, stderr={res.stderr.strip()}"
            )

    def restart_container(self):
        if not self._docker_inspect_running():
            self.start_container()
            return
        cmd = self._sudo(f"docker restart {shlex.quote(self.container_name)}")
        res = self._run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"重启 httpd 容器失败: {self.container_name}, stderr={res.stderr.strip()}"
            )
        self._wait_until_ready()

    def _wait_until_ready(self, timeout=None, interval=2):
        timeout = self.ready_timeout if timeout is None else int(timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._docker_inspect_running() and self.check_connection_alive():
                return True
            time.sleep(interval)
        raise RuntimeError(f"httpd 未在 {timeout}s 内就绪: {self.container_name} {self.server_url}")

    def check_connection_alive(self):
        cmd = (
            "curl --noproxy '*' -s -o /dev/null "
            f"-w '%{{http_code}}' --max-time 5 {shlex.quote(self.server_url)}"
        )
        res = self._run(cmd, check=False)
        if res.returncode != 0:
            return False
        try:
            code = int(res.stdout.strip())
        except Exception:
            return False
        return 200 <= code < 500

    def backup_config(self):
        self.start_container()
        self._ensure_local_parent(self.backup_path)
        self._ensure_local_parent(self.pristine_backup_path)
        if not os.path.exists(self.pristine_backup_path):
            cmd = self._sudo(
                f"docker cp {shlex.quote(self.container_name)}:{shlex.quote(self.config_file_path)} "
                f"{shlex.quote(self.pristine_backup_path)}"
            )
            self._run(cmd)
        self._run(self._sudo(f"cp {shlex.quote(self.pristine_backup_path)} {shlex.quote(self.backup_path)}"))
        self._suppress_access_log(self.backup_path)

    def _suppress_access_log(self, path):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = self._replace_or_append_directive(lines, "CustomLog", "/dev/null common")
        lines = self._replace_or_append_directive(lines, "TransferLog", "/dev/null")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def restore_config(self):
        if not os.path.exists(self.backup_path):
            return
        cmd = self._sudo(
            f"docker cp {shlex.quote(self.backup_path)} "
            f"{shlex.quote(self.container_name)}:{shlex.quote(self.config_file_path)}"
        )
        self._run(cmd)

    @staticmethod
    def _normalize_value(v):
        if isinstance(v, bool):
            return "On" if v else "Off"
        return str(v)

    @staticmethod
    def _replace_or_append_directive(lines, key, value):
        directive_re = re.compile(rf"^\s*{re.escape(key)}\b", re.IGNORECASE)
        replaced = False
        out = []
        for line in lines:
            if directive_re.match(line) and not line.lstrip().startswith("#"):
                out.append(f"{key} {value}\n")
                replaced = True
            else:
                out.append(line)
        if not replaced:
            out.append(f"{key} {value}\n")
        return out

    @staticmethod
    def _remove_directive(lines, key):
        directive_re = re.compile(rf"^\s*{re.escape(key)}\b", re.IGNORECASE)
        return [line for line in lines if not (directive_re.match(line) and not line.lstrip().startswith("#"))]

    def modify_config(self, knob_dict):
        self._ensure_local_parent(self.temp_path)
        if not os.path.exists(self.backup_path):
            raise RuntimeError(f"httpd backup config missing: {self.backup_path}")
        with open(self.backup_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for key, raw_val in knob_dict.items():
            if key == "__unset__":
                continue
            if is_unset_value(raw_val):
                lines = self._remove_directive(lines, key)
                continue
            val = self._normalize_value(raw_val)
            lines = self._replace_or_append_directive(lines, key, val)

        with open(self.temp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        cmd = self._sudo(
            f"docker cp {shlex.quote(self.temp_path)} "
            f"{shlex.quote(self.container_name)}:{shlex.quote(self.config_file_path)}"
        )
        self._run(cmd)

    def set_db_knob(self, config):
        try:
            self.backup_config()
            self.modify_config(config)
            self.restart_container()
            self.current_config = dict(config)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to set httpd knobs: {e}")
            return False

    def manage_database(self, action, dbname=None):
        if action == "start":
            self.start_container()
        elif action == "stop":
            self.stop_container()
        elif action == "restart":
            self.restart_container()
        return
