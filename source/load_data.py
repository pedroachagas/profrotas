import sys
import pandas as pd
import oracledb
from sqlalchemy import create_engine, text as sql_text
import config

def configure_oracle(version):
    # Configuring the module
    oracledb.version = version
    sys.modules["cx_Oracle"] = oracledb

def create_connection_string(dialect, username, password, host, port, service):
    # create the connection string
    return f"{dialect}://{username}:{password}@{host}:{port}/?service_name={service}"

def create_engine_from_string(config):
    # Variables
    DIALECT = config.DIALECT
    USERNAME = config.USERNAME
    PASSWORD = config.PASSWORD
    HOST = config.HOST
    PORT = config.PORT
    SERVICE = config.SERVICE

    connection_str = create_connection_string(DIALECT, USERNAME, PASSWORD, HOST, PORT, SERVICE)

    # create the engine
    print('Creating engine...')
    return create_engine(connection_str)

def execute_query(engine, query, csv_output):
    print(f"Executando query: {query}")
    df = pd.read_sql_query(sql=sql_text(query), con=engine.connect())
    df['data_consulta'] = pd.to_datetime('today')
    df.to_csv(csv_output, index=False)
    print(f"Dados da consulta salvos com sucesso em {csv_output}")

if __name__ == "__main__":
    configure_oracle("8.3.0")
    engine = create_engine_from_string(config)
    folder = "data/raw"

    queries_and_outputs = [
        ('SELECT * FROM "BOLEIA_SCHEMA"."V_COBRANCA"', f'{folder}/df_cobranca.csv'),
        ('SELECT * FROM "BOLEIA_SCHEMA"."V_FROTA"', f'{folder}/df_frota.csv')
    ]

    for query, output in queries_and_outputs:
        execute_query(engine, query, output)
