import json
import os
import shlex
import subprocess
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from systems.base import BaseSystemAdapter


def _safe_int(value, default):
    try:
        return int(float(value))
    except Exception:
        return default


class TomcatDB(BaseSystemAdapter):
    def __init__(self, args):
        self.args = args
        self.sys_name = "tomcat"
        self.host = args.get("host", "localhost")
        self.port = args.get("port", "8080")
        self.user = args.get("user", "")
        self.password = args.get("password", "")
        self.sudopassword = args.get("sudopassword", "")
        self.server_url = args.get("url", f"http://{self.host}:{self.port}/")
        self.container_name = args.get("container_name", args.get("dockername", "tomcat"))
        self.config_file_path = args.get("config_file_path", "/usr/local/tomcat/conf/server.xml")
        self.backup_path = args.get("backup_path", "./tempfiles/tomcat_backup.xml")
        self.temp_path = args.get("temp_config_file_path", "./tempfiles/tomcat_temp.xml")
        self.pristine_backup_path = f"{self.backup_path}.pristine"

        self.knobs_info = self.load_knobs(args["knob_config_file"], int(args["knob_num"]))
        self.default_knobs = self.build_default_knobs(self.knobs_info)
        self.current_config = dict(self.default_knobs)

    def _run(self, cmd):
        return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _run_text(self, cmd):
        return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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
        try:
            res = self._run_text(cmd)
            text = (res.stdout or "").strip()
            if text:
                parts.append({"source": "server_log", "text": text})
            if res.stderr and res.stderr.strip():
                parts.append({"source": "server_log_stderr", "text": res.stderr.strip()})
        except subprocess.CalledProcessError as e:
            stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", errors="ignore")
            stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", errors="ignore")
            if stdout.strip():
                parts.append({"source": "server_log", "text": stdout.strip()})
            if stderr.strip():
                parts.append({"source": "server_log_stderr", "text": stderr.strip()})
        return parts

    def _format_dependency_evidence(self):
        cfg = dict(getattr(self, "current_config", {}) or {})
        max_header = _safe_int(cfg.get("maxHttpHeaderSize"), 0)
        request_header = _safe_int(cfg.get("maxHttpRequestHeaderSize"), 0)
        response_header = _safe_int(cfg.get("maxHttpResponseHeaderSize"), 0)
        max_threads = _safe_int(cfg.get("maxThreads"), 0)
        processor_cache = _safe_int(cfg.get("processorCache"), 0)
        lines = ["TOMCAT_CONFIG_SNAPSHOT " + json.dumps(cfg, sort_keys=True)]

        if max_header >= 12000:
            lines.append(
                "TOMCAT_CONFIG_EVIDENCE tomcat_request_header_size_inherits_max_http_header_size active: "
                f"maxHttpHeaderSize={max_header} is active on the Connector; maxHttpRequestHeaderSize={request_header} controls request header capacity."
            )
        else:
            lines.append(
                "TOMCAT_CONFIG_EVIDENCE tomcat_request_header_size_inherits_max_http_header_size inactive: maxHttpHeaderSize is below the request-header dependency threshold."
            )

        if max_header >= 8192:
            lines.append(
                "TOMCAT_CONFIG_EVIDENCE tomcat_response_header_size_inherits_max_http_header_size active: "
                f"maxHttpHeaderSize={max_header} is active on the Connector; maxHttpResponseHeaderSize={response_header} controls response header capacity."
            )
        else:
            lines.append(
                "TOMCAT_CONFIG_EVIDENCE tomcat_response_header_size_inherits_max_http_header_size inactive: maxHttpHeaderSize is below the response-header dependency threshold."
            )

        if max_threads >= 200:
            lines.append(
                "TOMCAT_CONFIG_EVIDENCE tomcat_processor_cache_vs_max_threads active: "
                f"maxThreads={max_threads} exposes high request concurrency; processorCache={processor_cache} controls reusable processor capacity."
            )
        else:
            lines.append(
                "TOMCAT_CONFIG_EVIDENCE tomcat_processor_cache_vs_max_threads inactive: maxThreads is below the high-concurrency threshold."
            )
        return "\n".join(lines)

    def _sudo(self, cmd):
        if self.sudopassword:
            return f"echo {shlex.quote(str(self.sudopassword))} | sudo -S {cmd}"
        return cmd

    def backup_config(self):
        self._ensure_container_running()
        os.makedirs(os.path.dirname(self.backup_path) or ".", exist_ok=True)
        if not os.path.exists(self.pristine_backup_path):
            self._run(
                self._sudo(
                    f"docker cp {shlex.quote(self.container_name)}:{shlex.quote(self.config_file_path)} "
                    f"{shlex.quote(self.pristine_backup_path)}"
                )
            )
        self._run(self._sudo(f"cp {shlex.quote(self.pristine_backup_path)} {shlex.quote(self.backup_path)}"))

    def restore_config(self):
        self._run(self._sudo(f"docker cp {shlex.quote(self.backup_path)} {shlex.quote(self.container_name)}:{shlex.quote(self.config_file_path)}"))

    def modify_config(self, knob_dict):
        self._run(self._sudo(f"cp {shlex.quote(self.backup_path)} {shlex.quote(self.temp_path)}"))
        self._run(self._sudo(f"chmod 777 {shlex.quote(self.temp_path)}"))

        tree = ET.parse(self.temp_path)
        root = tree.getroot()
        connector = root.find(".//Connector")
        if connector is not None:
            unset_keys = knob_dict.get("__unset__", [])
            if isinstance(unset_keys, str):
                unset_keys = [unset_keys]
            for attr in unset_keys:
                connector.attrib.pop(str(attr), None)
            for attr, val in knob_dict.items():
                if attr == "__unset__":
                    continue
                connector.set(attr, str(val))
        tree.write(self.temp_path, encoding="utf-8", xml_declaration=True)

        self._run(self._sudo(f"chmod 644 {shlex.quote(self.temp_path)}"))
        self._run(self._sudo(f"docker cp {shlex.quote(self.temp_path)} {shlex.quote(self.container_name)}:{shlex.quote(self.config_file_path)}"))

    def get_config_value(self, attr):
        config_path = self.temp_path if os.path.exists(self.temp_path) else self.backup_path
        if not os.path.exists(config_path):
            return None
        tree = ET.parse(config_path)
        root = tree.getroot()
        connector = root.find(".//Connector")
        if connector is None:
            return None
        return connector.get(attr)

    def restart_container(self):
        self._run(self._sudo(f"docker restart {shlex.quote(self.container_name)}"))
        time.sleep(8)

    def _ensure_container_running(self):
        inspect_cmd = self._sudo(f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(self.container_name)}")
        try:
            res = self._run_text(inspect_cmd)
            running = res.stdout.strip().lower() == "true"
            if not running:
                self._run(self._sudo(f"docker start {shlex.quote(self.container_name)}"))
                time.sleep(3)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, (bytes, bytearray)) else str(e.stderr)
            raise RuntimeError(
                f"无法检查或启动容器 {self.container_name}。请确认容器存在且 Docker 可用。stderr={stderr}"
            ) from e

    def set_db_knob(self, config):
        try:
            self.backup_config()
            self.modify_config(config)
            self.restart_container()
            self.current_config = dict(config)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to set tomcat knobs: {e}")
            return False

    def check_connection_alive(self):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(self.server_url, timeout=5) as response:
                return 200 <= response.status < 500
        except urllib.error.HTTPError as e:
            return 200 <= int(e.code) < 500
        except Exception:
            return False

    def manage_database(self, action, dbname=None):
        return
