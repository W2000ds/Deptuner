from systems.httpddb import HttpdDB
from systems.mysqldb import MysqlDB
from systems.postgresqldb import PostgresqlDB
from systems.tomcatdb import TomcatDB
from systems.x265db import X265DB


SYSTEM_REGISTRY = {
    "mysql": MysqlDB,
    "postgresql": PostgresqlDB,
    "httpd": HttpdDB,
    "tomcat": TomcatDB,
    "x265": X265DB,
}


def create_target_system(args_db):
    sys_name = args_db["db"]
    if sys_name not in SYSTEM_REGISTRY:
        raise ValueError(f"Unsupported system: {sys_name}")
    return SYSTEM_REGISTRY[sys_name](args_db)
