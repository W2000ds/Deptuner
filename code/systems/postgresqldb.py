import json
import shlex
import time
import psycopg2
import subprocess
from utils.db_connector import PostgresqlConnector
from psycopg2 import sql
from systems.base import BaseSystemAdapter


class PostgresqlDB(BaseSystemAdapter):
    def __init__(self, args):
        self.args = args

        # PostgresSQL Info
        self.host = args['host']
        self.port = args['port']
        self.user = args['user']
        self.passwd = args['password']
        self.dbname = args['dbname']
        self.sudopassword = args.get('sudopassword', '')
        self.container_name = args.get('container_name') or f"{args['host']}_container"
        self.db_connector = PostgresqlConnector(self.host, self.port, self.user, self.passwd, self.dbname)

        # Postgresql Knobs
        self.knobs_info = self.load_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = self.build_default_knobs(self.knobs_info)


    def restart_container(self):
        """Restart the Docker container running Postgresql."""
        try:
            print(f"Restarting container: {self.container_name}")
            # Stop the container
            try:
                subprocess.run(["docker", "restart", self.container_name], check=True)
            except subprocess.CalledProcessError:
                if not self.sudopassword:
                    raise
                subprocess.run(
                    ["sudo", "-S", "docker", "restart", self.container_name],
                    input=f"{self.sudopassword}\n",
                    text=True,
                    check=True,
                )
            print(f"Container '{self.container_name}' restarted successfully.")

            if self.wait_until_postgresql_ready(timeout=60):
                # Close old connector
                if self.db_connector:
                    self.db_connector.close_db()

                # Reinitialize connector
                self.db_connector = PostgresqlConnector(self.host, self.port, self.user, self.passwd, self.dbname)
                print("[INFO] PostgreSQL connector re-initialized after restart.")
            else:
                print("[ERROR] PostgreSQL restart failed or too slow to respond.")
        except subprocess.CalledProcessError as e:
            print(f"Error restarting container '{self.container_name}': {e}")

    def restore_config(self):
        """Restore all tunable PostgreSQL knobs to repository defaults."""
        ok = self.set_db_knob(self.get_default_knobs())
        if not ok:
            raise RuntimeError("failed to restore PostgreSQL default knobs")

    def get_current_db_configurations(self):
        """Get the current configurations of PostgreSQL"""
        sql = "SHOW ALL;"
        configurations = self.db_connector.execute(sql)
        return {config[0]: config[1] for config in configurations}

    def set_db_knob(self, config):

        conn_count = self.check_active_connections()
        if conn_count != -1:
            print(f"[DEBUG] Current Postgresql connection count: {conn_count}")

        """Set the configs for PostgreSQL with connection check and container restart fallback."""
        if not self.check_connection_alive():
            print(f"[WARNING] Connection to {self.container_name} failed. Restarting container...")
            self.restart_container()
            time.sleep(5)  # wait for PostgreSQL to be ready
        try:
            """Set a configuration knob in PostgreSQL and verify the change."""
            self.db_connector.connect_db()
            # Enable autocommit mode for ALTER SYSTEM
            self.db_connector.conn.autocommit = True
            for knob_name, knob_value in config.items():
                sql = f"ALTER SYSTEM SET {knob_name} = '{knob_value}';"
                self.db_connector.execute(sql)
                # Reload configuration to apply changes
                self.db_connector.execute("SELECT pg_reload_conf();")
                print(f"[Knob Setting] Set {knob_name} = {knob_value}")

            self.db_connector.close_db()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to set knobs: {e}")
            return False

    def manage_database(self, action, dbname="testdb"):
        """Execute database management actions such as DROP DATABASE and CREATE DATABASE."""
        # [IMPORTANT]: Connect to a different database, like `postgres`; so that we can drop test db
        self.db_connector = PostgresqlConnector(self.host, self.port, self.user, self.passwd, "postgres")
        conn = self.db_connector.connect_db()
        conn.autocommit = True  # Enable autocommit to avoid transaction block errors

        try:
            cur = conn.cursor()
            if action == "drop":
                # Terminate all connections to `dbname`
                terminate_sql = f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{dbname}';"
                cur.execute(terminate_sql)
                # Drop the database if it exists
                cur.execute(f"DROP DATABASE IF EXISTS {dbname};")
                print(f"Database '{dbname}' dropped successfully.")

            elif action == "create":
                # Create the database if it does not exist
                cur.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(self.dbname)
                ))
                print(f"Database '{dbname}' created successfully.")

            elif action == "uuid":
                with psycopg2.connect(host=self.host, port=self.port, user=self.user, password=self.passwd, database=dbname) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cursor:
                        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
                        cursor.execute("""
                            DO $$ DECLARE r RECORD;
                            BEGIN
                                FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'sbtest%') LOOP
                                    EXECUTE format('
                                        ALTER TABLE %I ADD COLUMN uuid_id UUID DEFAULT uuid_generate_v4();
                                        UPDATE %I SET uuid_id = uuid_generate_v4();
                                        ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I_pkey;
                                        ALTER TABLE %I ADD PRIMARY KEY (uuid_id);
                                        ALTER TABLE %I DROP COLUMN id;
                                        ALTER TABLE %I RENAME COLUMN uuid_id TO id;
                                    ', r.tablename, r.tablename, r.tablename, r.tablename, r.tablename, r.tablename, r.tablename);
                                END LOOP;
                            END $$;
                        """)
                        print(f"All sbtest tables in {dbname} now use UUID as primary keys.")
            cur.close()
        except Exception as e:
            print(f"Error executing action '{action} {dbname}': {e}")
        finally:
            self.db_connector.close_db()




    def get_db_knob_value(self, knob_name):
        """Retrieve the current value of a specific PostgreSQL knob"""
        self.db_connector.connect_db()
        sql = f"SHOW {knob_name};"
        result = self.db_connector.execute(sql)
        # Debugging: Print the result to verify structure
        # print(f"Result of SHOW {knob_name}: {result}")
        # Access the first element directly if result is valid
        self.db_connector.close_db()
        return result[0][0] if result else None

    def collect_runtime_evidence(self, since=None, tail=200):
        parts = []
        snapshot = self._collect_db_snapshot()
        if snapshot:
            parts.append({"source": "db_state", "text": self._format_dependency_evidence(snapshot)})

        if self.container_name:
            since_clause = ""
            if since is not None:
                try:
                    since_clause = f" --since {shlex.quote(str(int(float(since))))}"
                except Exception:
                    since_clause = ""
            cmd = f"docker logs --tail {int(tail)}{since_clause} {shlex.quote(self.container_name)} 2>&1"
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                log_text = (result.stdout or "").strip()
                if log_text:
                    parts.append({"source": "db_log", "text": log_text})
                if result.stderr and result.stderr.strip():
                    parts.append({"source": "db_log_stderr", "text": result.stderr.strip()})
            except Exception:
                pass
        return parts

    def _collect_db_snapshot(self):
        snapshot = {}
        try:
            conn = self.db_connector.connect_db()
            cur = conn.cursor()
            for name in ("work_mem", "temp_buffers", "commit_delay", "commit_siblings"):
                cur.execute(f"SHOW {name};")
                row = cur.fetchone()
                snapshot[name] = row[0] if row else ""
            cur.execute(
                """
                SELECT temp_files, temp_bytes, numbackends
                FROM pg_stat_database
                WHERE datname = %s;
                """,
                (self.dbname,),
            )
            row = cur.fetchone()
            if row:
                snapshot["temp_files"] = row[0]
                snapshot["temp_bytes"] = row[1]
                snapshot["numbackends"] = row[2]
            cur.execute(
                """
                SELECT count(*)
                FROM pg_stat_activity
                WHERE backend_type = 'client backend'
                  AND datname = %s;
                """,
                (self.dbname,),
            )
            row = cur.fetchone()
            snapshot["client_backends"] = row[0] if row else 0
            self.db_connector.close_db()
        except Exception as exc:
            snapshot["evidence_error"] = str(exc)
        return snapshot

    @staticmethod
    def _format_dependency_evidence(snapshot):
        def int_value(name, default=0):
            try:
                raw = str(snapshot.get(name, default)).strip().split()[0]
                return int(float(raw))
            except Exception:
                return default

        temp_files = int_value("temp_files")
        temp_bytes = int_value("temp_bytes")
        commit_siblings = int_value("commit_siblings")
        client_backends = int_value("client_backends")

        lines = [
            "POSTGRES_VARIABLE_SNAPSHOT "
            + " ".join(f"{key}={snapshot.get(key, '')}" for key in sorted(snapshot.keys()))
        ]
        if temp_files > 0 or temp_bytes > 0:
            lines.append(
                "POSTGRES_DEPENDENCY_EVIDENCE postgres_work_mem_temp_buffers_low_pair active: "
                f"temporary file activity observed in pg_stat_database temp_files={temp_files} temp_bytes={temp_bytes}; "
                "sort/hash/temp-table execution can activate work_mem and temp_buffers."
            )
        else:
            lines.append(
                "POSTGRES_DEPENDENCY_EVIDENCE postgres_work_mem_temp_buffers_low_pair inactive: "
                "no temporary file activity observed in pg_stat_database; work_mem/temp_buffers dependency is not confirmed by this run."
            )

        if client_backends > commit_siblings:
            lines.append(
                "POSTGRES_DEPENDENCY_EVIDENCE postgres_commit_delay_always_wait active: "
                f"client_backends={client_backends} exceeds commit_siblings={commit_siblings}; commit_delay can be triggered by concurrent commit workload."
            )
        else:
            lines.append(
                "POSTGRES_RUNTIME_NOTE postgres_commit_delay_always_wait unobserved_after_workload: "
                f"post-run client_backends={client_backends} does not exceed commit_siblings={commit_siblings}; "
                "this snapshot is not a contradiction because commit_delay is triggered only during concurrent commits."
            )
        return "\n".join(lines)

    def check_active_connections(self, print_details=False):
        """
        Check current number of user connections to PostgreSQL, excluding internal background processes.
        """
        try:
            conn = self.db_connector.connect_db()
            cur = conn.cursor()
            cur.execute("""
                SELECT pid, usename, datname, state, client_addr, backend_type
                FROM pg_stat_activity
                WHERE backend_type = 'client backend'
                  AND datname = %s;
            """, (self.dbname,))
            rows = cur.fetchall()
            self.db_connector.close_db()

            conn_count = len(rows)
            if print_details:
                print(f"[INFO] Active user connections to '{self.dbname}': {conn_count}")
                for row in rows:
                    print(row)

            return conn_count

        except Exception as e:
            print(f"[ERROR] Failed to check PostgreSQL connections: {e}")
            return -1

    def wait_until_postgresql_ready(self, timeout=60):
        """Wait until PostgreSQL is ready to accept connections."""
        start = time.time()
        while time.time() - start < timeout:
            if self.check_connection_alive():
                print("[INFO] PostgreSQL is ready to accept connections.")
                return True
            time.sleep(5)
        print("[ERROR] PostgreSQL did not become ready in time.")
        return False

    def check_connection_alive(self):
        try:
            temp_connector = PostgresqlConnector(self.host, self.port, self.user, self.passwd, self.dbname)
            conn = temp_connector.connect_db()
            if conn:
                temp_connector.close_db()
                return True
        except Exception:
            return False
        return False
