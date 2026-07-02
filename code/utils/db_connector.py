import mysql.connector
import psycopg2
import os
from abc import ABC, abstractmethod
from mysql.connector import Error
import time
from mysql.connector import errorcode
from psycopg2 import sql, OperationalError
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


class DBConnector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def connect_db(self):
        pass

    @abstractmethod
    def close_db(self):
        pass

    @abstractmethod
    def fetch_results(self, sql, json=True):
        pass

    @abstractmethod
    def execute(self, sql):
        pass


class MysqlConnector(DBConnector):
    # Get env for test/develop; default setting is localhost
    def __init__(self,
                 host=os.getenv('DB_HOST', 'localhost'),
                 port=os.getenv('DB_PORT', 3306),
                 user=os.getenv('DB_USER', 'root'),
                 password=os.getenv('DB_PASSWORD', ''),
                 dbname=os.getenv('DB_NAME', 'testdb'),
                 flag=os.getenv('SSL_PRO', True)
                 ):
        super().__init__()
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        self.ssl_pro = flag
        self.conn = None
        self.cursor = None

    def connect_db(self):
        if self.conn:
            return self.conn
        last_err = None
        for attempt in range(1, 4):
            try:
                self.conn = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    passwd=self.password,
                    db=self.dbname,
                    port=self.port,
                    ssl_disabled=eval(self.ssl_pro),
                    connection_timeout=5,
                    autocommit=True
                )
                self.cursor = self.conn.cursor()
                return self.conn
            except mysql.connector.Error as err:
                last_err = err
                # if database doesn't exist, create it
                if err.errno == errorcode.ER_BAD_DB_ERROR:
                    print(f"Database '{self.dbname}' does not exist. Creating it now.")
                    self.conn = mysql.connector.connect(
                        host=self.host,
                        user=self.user,
                        passwd=self.password,
                        port=self.port,
                        ssl_disabled=eval(self.ssl_pro),
                        connection_timeout=5,
                        autocommit=True
                    )
                    self.cursor = self.conn.cursor()
                    self.cursor.execute(f"CREATE DATABASE {self.dbname}")
                    self.cursor.close()
                    # switch the new created database
                    self.conn.database = self.dbname
                    self.cursor = self.conn.cursor()
                    return self.conn
                if attempt < 3:
                    time.sleep(1)
                    continue
                raise
            except Exception as err:
                last_err = err
                if attempt < 3:
                    time.sleep(1)
                    continue
                raise

        if last_err:
            raise last_err
        return self.conn

    def close_db(self):
        try:
            if self.cursor:
                self.cursor.close()
        except:
            print("cursor close failed")
        try:
            if self.conn:
                self.conn.close()
        except:
            print("conn close failed")
        self.conn = None
        self.cursor = None

    def fetch_results(self, sql, json=True):
        results = False
        if self.conn:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            if json:
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
        return results

    def execute(self, sql, params=None):
        cursor = self.cursor
        cursor.execute(sql, params)
        result = cursor.fetchall()
        return result


class PostgresqlConnector(DBConnector):
    def __init__(self,
                 host=os.getenv('DB_HOST', 'localhost'),
                 port=os.getenv('DB_PORT', 5432),
                 user=os.getenv('DB_USER', 'root'),
                 password=os.getenv('DB_PASSWORD', '12345678'),
                 dbname=os.getenv('DB_NAME', 'testdb'),
                 ):
        super().__init__()
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        # self.conn = self.connect_db()
        # if self.conn:
        #     self.cursor = self.conn.cursor()
        self.conn = None
        self.cursor = None

    def connect_db(self):
        if self.conn:
            return self.conn
        try:
            # Attempt to connect to the specified database
            self.conn = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.dbname,
                port=self.port
            )
            self.cursor = self.conn.cursor()
            return self.conn
        except OperationalError as err:
            # If the database does not exist, create it
            if 'does not exist' in str(err):
                print(f"Database '{self.dbname}' does not exist. Creating it now.")
                self.create_database()  # Create the database
                # Re-attempt the connection to the newly created database
                self.conn = psycopg2.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.dbname,
                    port=self.port
                )
                self.cursor = self.conn.cursor()
                return self.conn
            else:
                raise  # If there's another error, raise it

    def create_database(self):
        """Create a new database outside of a transaction block."""
        # Connect to the default 'postgres' database
        conn = psycopg2.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port,
            database="postgres"  # Use a known existing database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)  # Set autocommit mode
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(self.dbname)
            ))
        conn.close()  # Close the connection after database creation

    def close_db(self):
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def fetch_results(self, sql, json=True):
        results = False
        if self.conn:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            if json:
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
        return results

    def execute(self, sql, params=None):
        try:
            self.connect_db()
            cursor = self.cursor
            cursor.execute(sql, params)
            if cursor.description is not None:
                return cursor.fetchall()
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Failed to execute SQL: {sql} -- {e}")
            return None
