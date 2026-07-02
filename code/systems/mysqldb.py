import subprocess
import time
import shlex
import os
import json
import configparser
import tempfile
import shutil
from utils.db_connector import DBConnector
from utils.db_connector import MysqlConnector
import re
from systems.base import BaseSystemAdapter


class MysqlDB(BaseSystemAdapter):
    def __init__(self, args):
        self.args = args

        # MySQL Info
        self.host = args['host']
        self.port = args['port']
        self.user = args['user']
        self.password = args.get('password', '')
        self.sudopassword = args['sudopassword']
        self.dbname = args['dbname']
        self.ssl_pro = args['ssl_pro']
        self.container_name = args['container_name']
        self.container_exec_user = args.get('container_exec_user', 'root')
        self.mysql_config_file = args.get('mysql_config_file', '/etc/my.cnf')
        self.mysql_config_section = args.get('mysql_config_section', 'mysqld')
        self.db_connector = MysqlConnector(self.host, self.port, self.user, self.password, self.dbname, self.ssl_pro)

        # MySQL Knobs
        self.knobs_info = self.load_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = self.build_default_knobs(self.knobs_info)
    def restart_container(self):
        """Restart the Docker container running MySQL."""
        try:
            print(f"Restarting container: {self.container_name}")

            subprocess.run(
            f"echo {self.password} | sudo -S docker-compose down -v",
            shell=True,
            )

            print("Docker Compose shut down successfully.")

            subprocess.run(
            f"echo {self.password} | sudo -S docker compose up -d",
            shell=True,
        )
            print(f"Container '{self.container_name}' restarted successfully.")

            if self.wait_until_mysql_ready(timeout=60):
                # Close old connector
                if self.db_connector:
                    self.db_connector.close_db()

                # Reinitialize connector
                self.db_connector = MysqlConnector(
                    self.host, self.port, self.user, self.password, self.dbname, self.ssl_pro
                )
                print("[INFO] MySQL connector re-initialized after restart.")
            else:
                print("[ERROR] MySQL restart failed or too slow to respond.")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Restarting container '{self.container_name}' failed: {e}")
    # TODO: need to be specified
    def get_current_db_configurations(self):
        """get the current configs of mysql"""
        sql = "SHOW VARIABLES WHERE Variable_name IN （'innodb_buffer_pool_size', 'innodb_log_file_size', ...);"  # ...indicates other variable
        configurations = self.db_connector.execute(sql)
        return {config[0]: config[1] for config in configurations}

    def set_db_knob(self, config):
        """set the configs for mysql"""
        conn_count = self.check_active_connections()
        if conn_count != -1:
            print(f"[DEBUG] Current MySQL connection count: {conn_count}")

        """Set the configs for MySQL with connection check and container restart fallback."""
        if not self.check_connection_alive():
            # In restricted environments, restart may require sudo/capabilities.
            print(f"[WARNING] Connection to {self.container_name} failed. Skip restart and continue.")
        conn = None
        success = True
        try:
            # getting connection before setting the value of knob; after that, close connection
            conn = self.db_connector.connect_db()
            for knob_name, knob_value in config.items():
                knob_meta = self.knobs_info.get(knob_name, {})
                if str(knob_meta.get("dynamic", "")).strip().lower() != "yes":
                    # Skip read-only knobs during tuning to avoid static-file+sandbox side effects.
                    print(f"[Knob Setting] Skip non-dynamic knob {knob_name}")
                    continue
                try:
                    sql = f"SET GLOBAL {knob_name} = {knob_value};"
                    self.db_connector.execute(sql)
                    print(f"[Knob Setting] Set {knob_name} = {knob_value}")
                except Exception as knob_error:
                    if not self._handle_read_only_knob(knob_name, knob_value, knob_error):
                        raise
        except Exception as e:
            print(f"[ERROR] Failed to set knobs: {e}")
            success = False
        finally:
            try:
                if conn:
                    self.db_connector.close_db()
            except Exception as close_error:
                print(f"[WARNING] Failed to close DB connection after knob setting: {close_error}")
        return success

    def _handle_read_only_knob(self, knob_name, knob_value, knob_error):
        """Handle read-only knob updates by editing MySQL config file."""
        error_msg = str(knob_error)
        read_only_patterns = [
            "is a read only variable",
            "variable is read-only",
            "is read only"
        ]
        if not any(pattern in error_msg.lower() for pattern in read_only_patterns):
            return False

        config_file = getattr(self, "mysql_config_file", None)
        section = getattr(self, "mysql_config_section", "mysqld")
        if not config_file:
            print(f"[WARNING] Cannot apply static update for {knob_name}: config file path missing.")
            return False

        if self.container_name:
            exists_in_container = self._container_path_exists(config_file)
            if not exists_in_container:
                print(f"[INFO] Config file '{config_file}' not found in container '{self.container_name}'. A new file will be created if necessary.")
        else:
            if not os.path.exists(config_file):
                print(f"[INFO] Config file '{config_file}' not found on host. A new file will be created if necessary.")

        print(f"[INFO] Knob '{knob_name}' is read-only. Falling back to static config update in {config_file}.")
        success = self.modify_mysql_config(config_file, section, knob_name, knob_value)
        return success

    def _build_sudo_command(self, command):
        if self.sudopassword:
            return f"echo {shlex.quote(str(self.sudopassword))} | sudo -S {command}"
        return command

    def _run_command(self, command, sudo=False, check=True):
        full_command = self._build_sudo_command(command) if sudo else command
        return subprocess.run(
            full_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def _chmod(self, target_path, mode):
        try:
            self._run_command(f"chmod {mode} {shlex.quote(target_path)}", sudo=True)
        except subprocess.CalledProcessError as error:
            stdout = error.stdout.decode().strip() if error.stdout else ""
            stderr = error.stderr.decode().strip() if error.stderr else ""
            print(f"[ERROR] Failed to chmod {target_path} to {mode}. stdout: {stdout}, stderr: {stderr}")
            raise

    def _docker_exec(self, inner_command, check=True, user=None):
        if not self.container_name:
            raise RuntimeError("Docker execution requested but container_name is not set.")
        exec_user = user if user is not None else getattr(self, "container_exec_user", None)
        user_flag = f"--user {shlex.quote(str(exec_user))} " if exec_user else ""
        command = (
            f"docker exec {user_flag}{shlex.quote(self.container_name)} "
            f"bash -lc {shlex.quote(inner_command)}"
        )
        return self._run_command(command, sudo=True, check=check)

    def _chmod_container(self, target_path, mode):
        try:
            inner_command = f"chmod {mode} {shlex.quote(target_path)}"
            self._docker_exec(inner_command)
        except subprocess.CalledProcessError as error:
            stdout = error.stdout.decode().strip() if error.stdout else ""
            stderr = error.stderr.decode().strip() if error.stderr else ""
            print(f"[ERROR] Failed to chmod {target_path} in container {self.container_name}. stdout: {stdout}, stderr: {stderr}")
            raise

    def _container_path_exists(self, target_path, directory=False):
        try:
            test_flag = "-d" if directory else "-f"
            inner_command = f"test {test_flag} {shlex.quote(target_path)}"
            self._docker_exec(inner_command)
            return True
        except subprocess.CalledProcessError:
            return False

    def _ensure_container_directory(self, dir_path):
        if not dir_path:
            return
        try:
            inner_command = f"mkdir -p {shlex.quote(dir_path)}"
            self._docker_exec(inner_command)
        except subprocess.CalledProcessError as error:
            stdout = error.stdout.decode().strip() if error.stdout else ""
            stderr = error.stderr.decode().strip() if error.stderr else ""
            print(f"[ERROR] Failed to ensure directory '{dir_path}' in container {self.container_name}. stdout: {stdout}, stderr: {stderr}")
            raise

    def _resolve_mysql_config_path(self, preferred_path):
        candidates = []
        if preferred_path:
            candidates.append(preferred_path)

        extra_candidates = getattr(self, "mysql_config_candidates", [])
        for path in extra_candidates:
            if path and path not in candidates:
                candidates.append(path)

        default_candidates = [
            "/etc/mysql/conf.d/tuning-overrides.cnf",
            "/etc/mysql/mysql.conf.d/mysqld.cnf",
            "/etc/mysql/my.cnf",
            "/etc/my.cnf",
        ]
        for path in default_candidates:
            if path not in candidates:
                candidates.append(path)

        for candidate in candidates:
            if self.container_name:
                if self._container_path_exists(candidate):
                    return candidate, False
            else:
                if os.path.exists(candidate):
                    return candidate, False

        # No existing file found; prepare to create the first candidate
        target_path = candidates[0]
        dir_path = os.path.dirname(target_path)
        if self.container_name:
            self._ensure_container_directory(dir_path)
        else:
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
        return target_path, True

    def modify_mysql_config(self, config_file, section, parameter, value):
        """Modify non-dynamic MySQL parameters via configuration file."""
        resolved_path, is_new = self._resolve_mysql_config_path(config_file)
        config_parser = configparser.RawConfigParser(allow_no_value=True, strict=False)
        config_parser.optionxform = str
        success = False
        temp_dir = None
        local_config_path = resolved_path
        try:
            if self.container_name:
                temp_dir = tempfile.mkdtemp(prefix="mysql-config-")
                local_config_path = os.path.join(temp_dir, os.path.basename(resolved_path))

                if not is_new:
                    self._chmod_container(resolved_path, "777")
                    self._run_command(
                        f"docker cp {shlex.quote(self.container_name)}:{shlex.quote(resolved_path)} {shlex.quote(local_config_path)}",
                        sudo=True,
                    )
                else:
                    with open(local_config_path, 'w') as new_config_handle:
                        new_config_handle.write(f"[{section}]\n")
            else:
                if is_new:
                    dir_path = os.path.dirname(resolved_path)
                    if dir_path and not os.path.exists(resolved_path):
                        os.makedirs(dir_path, exist_ok=True)
                    with open(resolved_path, 'w') as new_config_handle:
                        new_config_handle.write(f"[{section}]\n")
                self._chmod(resolved_path, "777")

            config_parser.read(local_config_path)

            if not config_parser.has_section(section):
                config_parser.add_section(section)

            config_parser.set(section, parameter, str(value))

            with open(local_config_path, 'w') as config_handle:
                config_parser.write(config_handle)

            if self.container_name:
                self._run_command(
                    f"docker cp {shlex.quote(local_config_path)} {shlex.quote(self.container_name)}:{shlex.quote(resolved_path)}",
                    sudo=True,
                )
                self._chmod_container(resolved_path, "644")
            else:
                self._chmod(resolved_path, "644")

            print(f"[INFO] Updated static parameter '{parameter}' to '{value}' in '{resolved_path}'.")
            success = True
        except Exception as error:
            print(f"[ERROR] Failed to modify MySQL config for '{parameter}': {error}")
            success = False
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

            if success:
                try:
                    self.restart_mysql_service()
                except Exception as restart_error:
                    print(f"[WARNING] Restart after static config update failed: {restart_error}")
                    success = False
        return success

    def restart_mysql_service(self):
        """Restart MySQL service or container to apply configuration changes."""
        try:
            if self.container_name:
                print(f"[INFO] Restarting MySQL container '{self.container_name}' to apply configuration changes.")
                result = self._run_command(f"docker restart {shlex.quote(self.container_name)}", sudo=True)
            else:
                print("[INFO] Restarting MySQL service to apply configuration changes.")
                result = self._run_command("systemctl restart mysql", sudo=True)
            stdout = result.stdout.decode().strip()
            stderr = result.stderr.decode().strip()
            if stdout:
                print(stdout)
            if stderr:
                print(stderr)
        except subprocess.CalledProcessError as error:
            stdout = error.stdout.decode().strip() if error.stdout else ""
            stderr = error.stderr.decode().strip() if error.stderr else ""
            print(f"[ERROR] Failed to restart MySQL service/container. stdout: {stdout}, stderr: {stderr}")
            raise

        if not self.wait_until_mysql_ready():
            raise RuntimeError("MySQL did not become ready after restarting service.")

    def manage_database(self, action, dbname="testdb"):

        """Manage MySQL database within Docker, with robust cleanup of both logical and physical remnants."""

        conn = self.db_connector.connect_db()
        try:
            cur = conn.cursor()

            if action == "drop":
                try:
                    cur.execute(f"DROP DATABASE IF EXISTS {dbname};")
                    print(f"[INFO] SQL DROP DATABASE '{dbname}' executed successfully.")
                except Exception as sql_drop_error:
                    print(f"[WARNING] SQL DROP DATABASE '{dbname}' failed: {sql_drop_error}")

                # check if dbname exists, if so, try to delete it
                check_cmd = f"echo {self.sudopassword} | sudo -S docker exec {self.container_name} bash -c 'test -d /var/lib/mysql/{dbname}'"
                if subprocess.run(check_cmd, shell=True).returncode == 0:
                    print(f"[INFO] Cleaning up Docker filesystem for '{dbname}'...")
                    docker_cmd = f"echo {self.sudopassword} | sudo -S docker exec {self.container_name} bash -c 'rm -rf /var/lib/mysql/{dbname}'"
                    result = subprocess.run(docker_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        print(f"[WARNING] Failed to delete physical files: {result.stderr.decode().strip()}")
                    else:
                        print(f"[INFO] Removed /var/lib/mysql/{dbname} from Docker.")
                else:
                    print(f"[INFO] No physical files found for '{dbname}'. Skipping cleanup.")

            if action == "create":
                try:
                    cur.execute(f"CREATE DATABASE IF NOT EXISTS {dbname};")
                    print(f"[INFO] Created database '{dbname}'.")
                except Exception as create_error:
                    print(f"[ERROR] Failed to create database '{dbname}': {create_error}")

            if action == "restart":
                self.restart_mysql(conn)

            cur.close()

        except Exception as e:
            print(f"[ERROR] manage_database('{action}', '{dbname}') failed: {e}")
        finally:
            self.db_connector.close_db()

    def restart_mysql(self, conn):
        """Restart MySQL server using SQL commands."""
        try:
            cur = conn.cursor()
            # Issue the SHUTDOWN command
            cur.execute("SHUTDOWN;")
            print("MySQL server shutdown successfully.")
            time.sleep(5)

            # Reconnect to the MySQL server
            self.db_connector = MysqlConnector(self.host, self.port, self.user, self.password, "mysql", self.ssl_pro)
            conn = self.db_connector.connect_db()
            print("MySQL server restarted successfully.")
        except Exception as e:
            print(f"Error restarting MySQL server: {e}")

    def get_db_knob_value(self, knob_name):

        conn = self.db_connector.connect_db()
        cur = conn.cursor()
        sql = "show variables like '{}';".format(knob_name)
        cur.execute(sql)
        result = cur.fetchall()
        self.db_connector.close_db()
        return result[0][1]

    def collect_runtime_evidence(self, since=None, tail=200):
        parts = []
        variables = self._show_variables(
            [
                "log_bin",
                "binlog_format",
                "binlog_row_image",
                "innodb_flush_log_at_trx_commit",
                "sync_binlog",
            ]
        )
        if variables:
            parts.append({"source": "db_state", "text": self._format_dependency_evidence(variables)})

        if self.container_name:
            since_clause = ""
            if since is not None:
                try:
                    since_clause = f" --since {shlex.quote(str(int(float(since))))}"
                except Exception:
                    since_clause = ""
            try:
                result = self._run_command(
                    f"docker logs --tail {int(tail)}{since_clause} {shlex.quote(self.container_name)} 2>&1",
                    sudo=True,
                    check=False,
                )
                log_text = (result.stdout or b"").decode("utf-8", errors="ignore").strip()
                if log_text:
                    parts.append({"source": "db_log", "text": log_text})
            except Exception:
                pass
        return parts

    def _show_variables(self, names):
        try:
            conn = self.db_connector.connect_db()
            cur = conn.cursor()
            placeholders = ",".join(["%s"] * len(names))
            cur.execute(f"SHOW VARIABLES WHERE Variable_name IN ({placeholders})", tuple(names))
            rows = cur.fetchall()
            self.db_connector.close_db()
            return {str(name): str(value) for name, value in rows}
        except Exception as exc:
            return {"evidence_error": str(exc)}

    @staticmethod
    def _format_dependency_evidence(variables):
        def val(name):
            return str(variables.get(name, "")).lower()

        log_bin_on = val("log_bin") in ("on", "1", "true")
        binlog_format = val("binlog_format")
        row_logging_active = log_bin_on and binlog_format in ("row", "mixed")
        flush_each_commit = val("innodb_flush_log_at_trx_commit") == "1"

        lines = [
            "MYSQL_VARIABLE_SNAPSHOT "
            + " ".join(f"{key}={variables.get(key, '')}" for key in sorted(variables.keys()))
        ]
        if log_bin_on and flush_each_commit:
            lines.append(
                "MYSQL_DEPENDENCY_EVIDENCE mysql_sync_binlog_with_innodb_flush_log active: "
                "binary logging is enabled and innodb_flush_log_at_trx_commit=1 puts commits on the InnoDB redo durability path; sync_binlog participates in binlog fsync durability."
            )
        else:
            reason = "binary logging disabled" if not log_bin_on else "innodb_flush_log_at_trx_commit is not 1"
            lines.append(
                "MYSQL_DEPENDENCY_EVIDENCE mysql_sync_binlog_with_innodb_flush_log inactive: "
                f"{reason}; sync_binlog is not confirmed as active for this dependency."
            )

        if row_logging_active and val("binlog_row_image") == "full":
            lines.append(
                "MYSQL_DEPENDENCY_EVIDENCE mysql_binlog_row_image_durability active: "
                "row-based binary logging is active and binlog_row_image=full controls the row image written to the binary log."
            )
        else:
            reason = "binary logging disabled" if not log_bin_on else f"binlog_format={variables.get('binlog_format', '')}"
            lines.append(
                "MYSQL_DEPENDENCY_EVIDENCE mysql_binlog_row_image_durability inactive: "
                f"{reason}; binlog_row_image is not confirmed as active for row-image durability."
            )
        return "\n".join(lines)

    def check_active_connections(self, print_details=False):
        """
        Check current number of active connections to MySQL.
        Returns: conn_count (int): number of active connections
        """
        try:
            conn = self.db_connector.connect_db()
            cur = conn.cursor()
            cur.execute("SHOW PROCESSLIST;")
            rows = cur.fetchall()
            self.db_connector.close_db()

            conn_count = len(rows)
            if print_details:
                print(f"[INFO] Active MySQL connections: {conn_count}")
                for row in rows:
                    print(row)

            return conn_count

        except Exception as e:
            print(f"[ERROR] Failed to check MySQL connections: {e}")
            return -1

    def wait_until_mysql_ready(self, timeout=60):
        """Wait until MySQL is ready to accept connections."""
        start = time.time()
        while time.time() - start < timeout:
            if self.check_connection_alive():
                print("[INFO] MySQL is ready to accept connections.")
                return True
            time.sleep(5)
        print("[ERROR] MySQL did not become ready in time.")
        return False

    def check_connection_alive(self):
        try:
            temp_connector = MysqlConnector(
                self.host, self.port, self.user, self.password, self.dbname, self.ssl_pro
            )
            conn = temp_connector.connect_db()
            if conn:
                temp_connector.close_db()
                return True
        except Exception:
            return False
        return False
