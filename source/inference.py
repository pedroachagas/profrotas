# Imports
import pandas as pd
import numpy as np

from tqdm import tqdm

import inflection
from unidecode import unidecode
from datetime import datetime, timedelta
from workadays import workdays as wd

import os
import shutil
import sys

import oracledb
from sqlalchemy import create_engine, text as sql_text
import config

from sklearn.linear_model import LinearRegression
# Funções Auxiliares
def print_date_ranges(df):
    for col in df.select_dtypes(include=['datetime']):
        print(f"{col}: {df[col].min().date()} até {df[col].max().date()}")
## Load data
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
    return df


def format_cnpj(input_string):
    input_string = list(input_string)  # Convert string to list because strings are immutable in Python

    if len(input_string) != 14:
        return 'Input string must have exactly 14 characters'
    
    input_string.insert(2, '.')
    input_string.insert(6, '.')
    input_string.insert(10, '/')
    input_string.insert(15, '-')

    return ''.join(input_string)

def raiz(cnpj):
    # Retorna uma string contendo apenas os primeiros 8 dígitos do CNPJ
    return str(cnpj)[:-8]

def preprocess_cnpj(df):
    cnpj_col = df.filter(like='cnpj').columns[0]
    df[cnpj_col] = df[cnpj_col].astype(str).str.zfill(14)
    df['cnpj_raiz'] = df[cnpj_col].apply(format_cnpj).apply(raiz)
    return df

def dias_uteis_entre_datas(df, start_date_col, end_date_col):
    # Função auxiliar que calcula os dias úteis entre as datas de início e fim de uma linha do dataframe
    def dias_uteis(row):
        if pd.isnull(row[start_date_col]) or pd.isnull(row[end_date_col]):
            return None
        return wd.networkdays(row[end_date_col].date(), row[start_date_col].date())

    # Aplica a função dias_uteis a cada linha do dataframe e retorna uma série com o número de dias úteis calculado
    df["dias_uteis"] = df.apply(dias_uteis, axis=1)

    return df["dias_uteis"]


def filter_unpaid_after_first(df):
    df_vencimentos = (
        df[df.target == "vencido"]
        .groupby(["cnpj_raiz"])["vencimento"]
        .min()
        .reset_index()
        .rename(columns={"vencimento": "data_primeiro_vencimento"})
    )

    df = df.merge(df_vencimentos, on="cnpj_raiz", how="left")
    
    base_df = df.loc[
        df.data_primeiro_vencimento.isna()
        | (df.vencimento <= df.data_primeiro_vencimento),:
    ].drop('data_primeiro_vencimento', axis=1)

    return base_df


def remove_same_day_duplicates(df_base):
    target_encoding = {
        "pgto_adiantado": 0,
        "pgto_no_dia": 1,
        "pgto_atrasado_menos_5": 2,
        "pgto_atrasado_mais_5": 3,
        "vencido": 4,
    }

    df_base.target = df_base.target.map(target_encoding)
    df_base = df_base.sort_values(
        by=["cnpj_raiz", "vencimento", "target", "dias_atraso", "valor_total"],
        ascending=[True, True, False, False, False],
    )
    df_base = df_base.drop_duplicates(subset=["cnpj_raiz", "vencimento"], keep="first")

    reverse_target_encoding = {value: key for key, value in target_encoding.items()}
    df_base.target = df_base.target.map(reverse_target_encoding)

    return df_base


def preprocess_dates_cobranca(df_cobranca):
    df_cobranca.data_pagamento = pd.to_datetime(df_cobranca.data_pagamento.dt.date)
    df_cobranca['data_lancamento'] = df_cobranca.fim_periodo.dt.date
    df_cobranca.data_lancamento = pd.to_datetime(df_cobranca.data_lancamento)
    df_cobranca.data_consulta = pd.to_datetime(df_cobranca.data_consulta)
    df_cobranca.vencimento = pd.to_datetime(df_cobranca.vencimento.astype(str).str.replace('2102','2022'))
    df_cobranca['data_atraso'] = df_cobranca.vencimento.apply(lambda x: np.nan if pd.isnull(x) else wd.workdays(x, 6))
    df_cobranca.data_atraso = pd.to_datetime(df_cobranca.data_atraso)

    return df_cobranca


def preprocess_valor_cobranca(df_cobranca):
    df_cobranca = df_cobranca.rename(columns={'valor_total_r$':'valor_total'})
    df_cobranca.valor_total = df_cobranca.valor_total.astype(float)

    return df_cobranca


def preprocess_target_cobranca(df_cobranca):
    df_cobranca['dias_atraso'] = dias_uteis_entre_datas(df_cobranca, 'data_pagamento', 'vencimento')
    df_cobranca["target"] = df_cobranca["dias_atraso"].apply(
        lambda x: "pgto_adiantado"
        if x < 0
        else "pgto_no_dia"
        if x == 0
        else "pgto_atrasado_menos_5"
        if x <= 5
        else "pgto_atrasado_mais_5"
        if x > 5
        else "vencido"
    )

    return df_cobranca


def preprocess_public_cobranca(df_cobranca):
    cond1 = df_cobranca.modalidade_da_frota == 1  
    cond2 = df_cobranca.vencimento.notna()  
    cond3 = df_cobranca.vencimento > df_cobranca.data_lancamento  
    condicoes = cond1 & cond2 & cond3
    df_cobranca_filtered = (df_cobranca[condicoes]
                            .pipe(filter_unpaid_after_first)
                            .pipe(remove_same_day_duplicates))
    
    return df_cobranca_filtered


def select_relevant_columns(df_cobranca):
    return df_cobranca[[
        'cnpj_raiz',
        'vencimento',
        'data_pagamento',
        'data_atraso',
        'data_lancamento',
        'valor_total',
        'transacoes',
        'dias_atraso',
        'target'
        ]].copy()


def preprocess_cobranca(df):
    print("Começando processamento da base COBRANCAS...")
    df_processed = (
            df.pipe(preprocess_cnpj)
                .pipe(preprocess_dates_cobranca)
                .pipe(preprocess_valor_cobranca)
                .pipe(preprocess_target_cobranca)
                .pipe(preprocess_public_cobranca)
                .pipe(select_relevant_columns)
    )
    print("Processamento concluído.")
    df_processed.to_csv('data/inference/df_base.csv', index=False)
    return df_processed


def transform_dataframe(df):
    # Ensure that all columns are present in the dataframe
    required_columns = [
        "cnpj_raiz", "vencimento", "data_pagamento", 
        "data_atraso", "data_lancamento", "valor_total", 
        "transacoes", "dias_atraso", "target"
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
    
    # Transform each column to the desired data type
    df["cnpj_raiz"] = df["cnpj_raiz"].astype('object')
    # Check if conversion to datetime is possible, else fill NaNs
    for date_column in ["vencimento", "data_pagamento", "data_atraso", "data_lancamento"]:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df[date_column].replace({np.nan:pd.NaT}, inplace=True)

    df["valor_total"] = df["valor_total"].astype('float64')
    df["transacoes"] = df["transacoes"].astype('int64')
    df["dias_atraso"] = df["dias_atraso"].astype('float64')
    df["target"] = df["target"].astype('object')

    return df


# Função para criar novas colunas com o número de linhas filtradas
def create_num_cols(df, df_last):
    # Função para contar o número de linhas filtradas
    def filtered_df(cnpj, ref_date, periodo, target, df):
        # Cria máscaras booleanas para cada condição de filtragem
        mask_cnpj = df.index.get_level_values(0) == cnpj
        mask_vencimento = df.data_atraso < ref_date                                                       # (para a qual é calculado o histórico) menos 6 dias

        # Combina as máscaras usando o operador &
        combined_mask = mask_cnpj & mask_vencimento

        # Aplica a restrição de período, se fornecida
        if periodo is not None:
            mask_periodo = df.index.get_level_values(1) >= pd.Timestamp(
                ref_date
            ) - pd.to_timedelta(periodo, unit="D")
            combined_mask = combined_mask & mask_periodo

        # Aplica a restrição de target, se fornecida
        if target is not None:
            mask_target = df["target"] == target
            combined_mask = combined_mask & mask_target

        # Aplica a máscara combinada ao dataframe
        filtered_df = df[combined_mask]
        return len(filtered_df)

    # Cria uma lista de períodos e targets para serem usados
    df = df.copy()
    intervalos_de_tempo = [30, 60, 90, 120]
    targets = [
        "pgto_no_dia",
        "pgto_atrasado_mais_5",
        "pgto_atrasado_menos_5",
        "pgto_adiantado",
    ]

    # Reseta e reconfigura o índice do dataframe antes do loop
    df.reset_index(inplace=True)
    df.set_index(["cnpj_raiz", "vencimento"], inplace=True)
    df.sort_index(inplace=True)

    if 'cnpj_raiz' not in df_last.index.names or 'vencimento' not in df_last.index.names:
        df_last.set_index(["cnpj_raiz", "vencimento"], inplace=True)
        df_last.sort_index(inplace=True)

    total_entries = len(df_last.index)
    progress_bar = tqdm(desc='Criando features de contagem de cobranças',total=total_entries * len(targets) * (len(intervalos_de_tempo) + 1))

    # Cria uma nova coluna para cada target e para cada intervalo de tempo
    df_last["num_cobrancas"] = 0
    for i in df_last.index:
        # Usa a função filtered_df para obter o número de linhas filtradas
        df_last.loc[i, "num_cobrancas"] = filtered_df(
            i[0], df_last.loc[i, "data_lancamento"], None, None, df
        )
    for target in targets:
        total_col = f"num_{target}_total"
        df_last[total_col] = 0
        for i in df_last.index:
            # Usa a função filtered_df para obter o número de linhas filtradas
            df_last.loc[i, total_col] = filtered_df(
                i[0], df_last.loc[i, "data_lancamento"], None, target, df
            )
            
            progress_bar.update(1)
            
        for periodo in intervalos_de_tempo:
            col_name = f"num_{target}_ult_{periodo}_dias"
            df_last[col_name] = 0
            for i in df_last.index:
                # Usa a função filtered_df para obter o número de linhas filtradas
                df_last.loc[i, col_name] = filtered_df(
                    i[0], df_last.loc[i, "data_lancamento"], periodo, target, df
                )

                progress_bar.update(1)

    progress_bar.close()

    # Reorganiza o índice e remove a coluna "index"
    return df_last.sort_index().reset_index()


def add_prop_columns(df):
    # Define as listas de intervalos de tempo e targets
    intervals = [30, 60, 90, 120]
    targets = [
        "pgto_no_dia",
        "pgto_atrasado_mais_5",
        "pgto_atrasado_menos_5",
        "pgto_adiantado",
    ]

    # Define o valor a ser usado em caso de divisão por zero
    fill_value = np.nan

    # Cria uma nova coluna de proporção para cada combinação de target e intervalo
    for target in targets:
        for interval in intervals:
            num_col = f"num_{target}_ult_{interval}_dias"
            prop_col = f"prop_{target}_ult_{interval}_dias"
            df[prop_col] = fill_value
            # Calcula o número total de pagamentos no intervalo de tempo
            total_interval_payments = (
                df[f"num_pgto_no_dia_ult_{interval}_dias"]
                + df[f"num_pgto_atrasado_mais_5_ult_{interval}_dias"]
                + df[f"num_pgto_atrasado_menos_5_ult_{interval}_dias"]
                + df[f"num_pgto_adiantado_ult_{interval}_dias"]
            )

            # Calcula a proporção e atribui à nova coluna
            df[prop_col] = df[num_col].divide(
                total_interval_payments, fill_value=fill_value
            )

    # Cria uma nova coluna de proporção para cada target com o número total de pagamentos
    for target in targets:
        total_num_col = f"num_{target}_total"
        total_prop_col = f"prop_{target}_total"
        df[total_prop_col] = fill_value
        # Calcula o número total de pagamentos para todos os targets
        total_payments = (
            df["num_pgto_no_dia_total"]
            + df["num_pgto_atrasado_mais_5_total"]
            + df["num_pgto_atrasado_menos_5_total"]
            + df["num_pgto_adiantado_total"]
        )

        # Calcula a proporção e atribui à nova coluna
        df[total_prop_col] = df[total_num_col].divide(
            total_payments, fill_value=fill_value
        )

    return df


# Função para adicionar colunas com informações de atraso baseadas em dias
def add_days_columns(df):
    # Lista de intervalos de tempo usados no loop
    TIME_INTERVALS = [30, 60, 90, 120]
    # Valor de preenchimento padrão
    FILL_VALUE = 0

    # Função para calcular estatísticas básicas em um conjunto de registros
    def calculate_statistics(records):
        mean = records["dias_atraso"].mean()
        median = records["dias_atraso"].median()
        std = records["dias_atraso"].std()
        if len(records) > 1:
            X = np.arange(len(records)).reshape(-1, 1)
            y = records["dias_atraso"].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0][0]
        else:
            slope = FILL_VALUE
        return mean, median, std, slope

    # Converte a coluna de vencimento para o tipo datetime
    df["vencimento"] = pd.to_datetime(df["vencimento"])

    # Cria novas colunas para cada intervalo de tempo e cada estatística a ser calculada
    for periodo in TIME_INTERVALS:
        for prefix in [
            "media_atraso_ult",
            "mediana_atraso_ult",
            "desvio_padrao_atraso_ult",
            "inclinacao_dias_atraso_ult",
        ]:
            column_name = f"{prefix}_{periodo}_dias"
            df[column_name] = FILL_VALUE

    # Cria novas colunas para o total de cada estatística
    for prefix in [
        "media_total_atraso",
        "mediana_total_atraso",
        "desvio_padrao_total_atraso",
        "inclinacao_dias_atraso_total",
    ]:
        df[prefix] = FILL_VALUE

    # Itera pelos grupos de CNPJ raiz no dataframe
    for cnpj_raiz, group in df.groupby("cnpj_raiz"):
        past_records = group.sort_values("vencimento")

        # Itera pelas linhas de registros passados e adiciona as informações de atraso para cada uma delas
        for i in range(len(past_records)):
            current_row = past_records.iloc[i]
            date_cutoffs = [
                current_row["vencimento"] - pd.Timedelta(days=periodo)
                for periodo in TIME_INTERVALS
            ]
            past_records_subset = past_records[
                past_records["vencimento"] < current_row["vencimento"]
            ]

            # Calcula as estatísticas para o total e para cada intervalo de tempo
            if not past_records_subset.empty:
                total_mean, total_median, total_std, total_slope = calculate_statistics(
                    past_records_subset
                )
                df.loc[current_row.name, "media_total_atraso"] = total_mean
                df.loc[current_row.name, "mediana_total_atraso"] = total_median
                df.loc[current_row.name, "desvio_padrao_total_atraso"] = total_std
                df.loc[current_row.name, "inclinacao_dias_atraso_total"] = total_slope

                for periodo, date_cutoff in zip(TIME_INTERVALS, date_cutoffs):
                    period_records = past_records_subset[
                        past_records_subset["vencimento"] >= date_cutoff
                    ]
                    if not period_records.empty:
                        (
                            interval_mean,
                            interval_median,
                            interval_std,
                            interval_slope,
                        ) = calculate_statistics(period_records)
                        # Adiciona as estatísticas para cada intervalo de tempo
                        column_name = f"media_atraso_ult_{periodo}_dias"
                        df.loc[current_row.name, column_name] = interval_mean
                        column_name = f"mediana_atraso_ult_{periodo}_dias"
                        df.loc[current_row.name, column_name] = interval_median
                        column_name = f"desvio_padrao_atraso_ult_{periodo}_dias"
                        df.loc[current_row.name, column_name] = interval_std
                        column_name = f"inclinacao_dias_atraso_ult_{periodo}_dias"
                        df.loc[current_row.name, column_name] = interval_slope

        # Preenche valores NaN com o valor padrão
        df.fillna(FILL_VALUE, inplace=True)

    return df


def add_vigency_information(df):
    # Add a new column with the start date of the vigency for each CNPJ root
    df["data_inicio_vigencia"] = df.groupby("cnpj_raiz")["vencimento"].transform("min")

    # Add a new column with the number of days from the start of the vigency to the due date
    df["dias_vigencia"] = (df["vencimento"] - df["data_inicio_vigencia"]).dt.days

    # Return the columns 'vencimento', 'data_inicio_vigencia', and 'dias_vigencia'
    return df.drop("data_inicio_vigencia", axis=1)


# Função para adicionar novas colunas de razão a um dataframe
def add_razao_cols(df):
    # Define o valor de preenchimento padrão para divisões por zero
    fill_value = 1

    # Define as colunas de número de pagamentos atrasados/adiantados e os intervalos de tempo a serem usados
    razao_cols = [
        (
            "num_pgto_atrasado_mais_5_ult_30_dias",
            "num_pgto_atrasado_mais_5_ult_60_dias",
            "razao_num_pgto_atrasado_mais_5_30_60",
        ),
        (
            "num_pgto_atrasado_mais_5_ult_30_dias",
            "num_pgto_atrasado_mais_5_ult_90_dias",
            "razao_num_pgto_atrasado_mais_5_30_90",
        ),
        (
            "num_pgto_atrasado_mais_5_ult_30_dias",
            "num_pgto_atrasado_mais_5_ult_120_dias",
            "razao_num_pgto_atrasado_mais_5_30_120",
        ),
        (
            "num_pgto_atrasado_menos_5_ult_30_dias",
            "num_pgto_atrasado_menos_5_ult_60_dias",
            "razao_num_pgto_atrasado_menos_5_30_60",
        ),
        (
            "num_pgto_atrasado_menos_5_ult_30_dias",
            "num_pgto_atrasado_menos_5_ult_90_dias",
            "razao_num_pgto_atrasado_menos_5_30_90",
        ),
        (
            "num_pgto_atrasado_menos_5_ult_30_dias",
            "num_pgto_atrasado_menos_5_ult_120_dias",
            "razao_num_pgto_atrasado_menos_5_30_120",
        ),
        (
            "num_pgto_no_dia_ult_30_dias",
            "num_pgto_no_dia_ult_60_dias",
            "razao_num_pgto_no_dia_30_60",
        ),
        (
            "num_pgto_no_dia_ult_30_dias",
            "num_pgto_no_dia_ult_90_dias",
            "razao_num_pgto_no_dia_30_90",
        ),
        (
            "num_pgto_no_dia_ult_30_dias",
            "num_pgto_no_dia_ult_120_dias",
            "razao_num_pgto_no_dia_30_120",
        ),
        (
            "num_pgto_adiantado_ult_30_dias",
            "num_pgto_adiantado_ult_60_dias",
            "razao_num_pgto_adiantado_30_60",
        ),
        (
            "num_pgto_adiantado_ult_30_dias",
            "num_pgto_adiantado_ult_90_dias",
            "razao_num_pgto_adiantado_30_90",
        ),
        (
            "num_pgto_adiantado_ult_30_dias",
            "num_pgto_adiantado_ult_120_dias",
            "razao_num_pgto_adiantado_30_120",
        ),
    ]

    # Cria as novas colunas de razão para cada combinação de colunas de número de pagamentos
    df = df.copy()
    for num_30, num_longer, razao_col in razao_cols:
        df[razao_col] = df[num_30] / df[num_longer].replace(0, fill_value)

    return df

# Função para criar novas colunas com o valor total das linhas filtradas
def create_value_columns(df, df_last):
    # Função para calcular o valor total das linhas filtradas
    def filtered_value(cnpj, ref_date, periodo, target, df):

        # Cria máscaras booleanas para cada condição de filtragem
        mask_cnpj = df.index.get_level_values(0) == cnpj
        mask_vencimento = df.data_atraso < ref_date
        mask_target = df["target"] == target

        # Combina as máscaras usando o operador &
        combined_mask = mask_cnpj & mask_vencimento & mask_target

        # Aplica a restrição de período, se fornecida
        if periodo is not None:
            start_date = pd.to_datetime(ref_date) - pd.Timedelta(days=periodo)
            mask_periodo = pd.to_datetime(df.data_atraso) >= start_date
            combined_mask = combined_mask & mask_periodo

        # Aplica a máscara combinada ao dataframe
        filtered_df = df[combined_mask]
        return filtered_df["valor_total"].sum()

    # Cria uma lista de períodos e targets para serem usados
    intervalos_de_tempo = [30, 60, 90, 120]
    targets = [
        "pgto_no_dia",
        "pgto_atrasado_mais_5",
        "pgto_atrasado_menos_5",
        "pgto_adiantado",
    ]

    # Reseta e reconfigura o índice do dataframe antes do loop
    df = df.copy()
    df.reset_index(inplace=True)
    df.set_index(["cnpj_raiz", "vencimento"], inplace=True)
    df.sort_index(inplace=True)

    if 'cnpj_raiz' not in df_last.index.names or 'vencimento' not in df_last.index.names:
        df_last.set_index(["cnpj_raiz", "vencimento"], inplace=True)
        df_last.sort_index(inplace=True)

    total_entries = len(df_last.index)
    progress_bar = tqdm(desc='Criando features de valores das cobranças',total=total_entries * len(targets) * (len(intervalos_de_tempo) + 1))

    # Cria uma nova coluna para cada target e para cada intervalo de tempo
    for target in targets:
        total_col = f"valor_{target}_total"
        df_last[total_col] = 0
        for i in df_last.index:
            # Usa a função filtered_value para obter o número de linhas filtradas
            df_last.loc[i, total_col] = filtered_value(
                i[0], df_last.loc[i, "data_lancamento"], None, target, df
            )

            progress_bar.update(1)

        for periodo in intervalos_de_tempo:
            col_name = f"valor_{target}_ult_{periodo}_dias"
            df_last[col_name] = 0
            for i in df_last.index:
                # Usa a função filtered_value para obter o número de linhas filtradas
                df_last.loc[i, col_name] = filtered_value(
                    i[0], df_last.loc[i, "data_lancamento"], periodo, target, df
                )

                progress_bar.update(1)

    progress_bar.close()

    # Reorganiza o índice e remove a coluna "index"
    return df_last.sort_index().reset_index()

# Função para adicionar novas colunas de razão de valores
def add_razao_valor_cols(df):
    # Valor a ser usado no lugar de 0 na divisão
    fill_value = 1

    # Lista de tuplas com as colunas de valor e as colunas de razão a serem criadas
    razao_cols = [
        (
            "valor_pgto_atrasado_mais_5_ult_30_dias",
            "valor_pgto_atrasado_mais_5_ult_60_dias",
            "razao_valor_pgto_atrasado_mais_5_30_60",
        ),
        (
            "valor_pgto_atrasado_mais_5_ult_30_dias",
            "valor_pgto_atrasado_mais_5_ult_90_dias",
            "razao_valor_pgto_atrasado_mais_5_30_90",
        ),
        (
            "valor_pgto_atrasado_mais_5_ult_30_dias",
            "valor_pgto_atrasado_mais_5_ult_120_dias",
            "razao_valor_pgto_atrasado_mais_5_30_120",
        ),
        (
            "valor_pgto_atrasado_menos_5_ult_30_dias",
            "valor_pgto_atrasado_menos_5_ult_60_dias",
            "razao_valor_pgto_atrasado_menos_5_30_60",
        ),
        (
            "valor_pgto_atrasado_menos_5_ult_30_dias",
            "valor_pgto_atrasado_menos_5_ult_90_dias",
            "razao_valor_pgto_atrasado_menos_5_30_90",
        ),
        (
            "valor_pgto_atrasado_menos_5_ult_30_dias",
            "valor_pgto_atrasado_menos_5_ult_120_dias",
            "razao_valor_pgto_atrasado_menos_5_30_120",
        ),
        (
            "valor_pgto_no_dia_ult_30_dias",
            "valor_pgto_no_dia_ult_60_dias",
            "razao_valor_pgto_no_dia_30_60",
        ),
        (
            "valor_pgto_no_dia_ult_30_dias",
            "valor_pgto_no_dia_ult_90_dias",
            "razao_valor_pgto_no_dia_30_90",
        ),
        (
            "valor_pgto_no_dia_ult_30_dias",
            "valor_pgto_no_dia_ult_120_dias",
            "razao_valor_pgto_no_dia_30_120",
        ),
        (
            "valor_pgto_adiantado_ult_30_dias",
            "valor_pgto_adiantado_ult_60_dias",
            "razao_valor_pgto_adiantado_30_60",
        ),
        (
            "valor_pgto_adiantado_ult_30_dias",
            "valor_pgto_adiantado_ult_90_dias",
            "razao_valor_pgto_adiantado_30_90",
        ),
        (
            "valor_pgto_adiantado_ult_30_dias",
            "valor_pgto_adiantado_ult_120_dias",
            "razao_valor_pgto_adiantado_30_120",
        ),
    ]

    # Cria novas colunas de razão de valor para cada tupla de colunas de valor
    for num_30, num_longer, razao_col in razao_cols:
        df[razao_col] = df[num_30] / df[num_longer].replace(0, fill_value)

    return df


# Função para criar novas colunas com o valor total das linhas filtradas
def create_trans_columns(df, df_last):
    # Função para calcular o valor total das linhas filtradas
    def filtered_trans(cnpj, ref_date, periodo, target, df):
        # Cria máscaras booleanas para cada condição de filtragem
        mask_cnpj = df.index.get_level_values(0) == cnpj
        mask_vencimento = df.data_atraso < ref_date
        mask_target = df["target"] == target

        # Combina as máscaras usando o operador &
        combined_mask = mask_cnpj & mask_vencimento & mask_target

        # Aplica a restrição de período, se fornecida
        if periodo is not None:
            start_date = pd.to_datetime(ref_date) - pd.Timedelta(days=periodo)
            mask_periodo = pd.to_datetime(df.data_atraso) >= start_date
            combined_mask = combined_mask & mask_periodo

        # Aplica a máscara combinada ao dataframe
        filtered_df = df[combined_mask]
        return filtered_df["transacoes"].sum()

    # Cria uma lista de períodos e targets para serem usados
    df = df.copy()
    intervalos_de_tempo = [30, 60, 90, 120]
    targets = [
        "pgto_no_dia",
        "pgto_atrasado_mais_5",
        "pgto_atrasado_menos_5",
        "pgto_adiantado",
    ]

    # Reseta e reconfigura o índice do dataframe antes do loop
    df.reset_index(inplace=True)
    df.set_index(["cnpj_raiz", "vencimento"], inplace=True)
    df.sort_index(inplace=True)

    if 'cnpj_raiz' not in df_last.index.names or 'vencimento' not in df_last.index.names:
        df_last.set_index(["cnpj_raiz", "vencimento"], inplace=True)
        df_last.sort_index(inplace=True)

    total_entries = len(df_last.index)
    progress_bar = tqdm(desc='Criando features de número de transações',total=total_entries * len(targets) * (len(intervalos_de_tempo) + 1))

    # Cria uma nova coluna para cada target e para cada intervalo de tempo
    for target in targets:
        total_col = f"num_transacoes_{target}_total"
        df_last[total_col] = 0
        for i in df_last.index:
            # Usa a função filtered_df para obter o número de linhas filtradas
            df_last.loc[i, total_col] = filtered_trans(
                i[0], df_last.loc[i, "data_lancamento"], None, target, df
            )

            progress_bar.update(1)
        for periodo in intervalos_de_tempo:
            col_name = f"num_transacoes_{target}_ult_{periodo}_dias"
            df_last[col_name] = 0
            for i in df_last.index:
                # Usa a função filtered_df para obter o número de linhas filtradas
                df_last.loc[i, col_name] = filtered_trans(
                    i[0], df_last.loc[i, "data_lancamento"], periodo, target, df
                )

                progress_bar.update(1)

    progress_bar.close()

    # Reorganiza o índice e remove a coluna "index"
    return df_last.sort_index().reset_index()


def add_razao_transacoes_cols(df):
    # Valor padrão para substituir divisões por zero
    fill_value = 1

    # Lista de colunas para cálculo da razão
    razao_cols = [
        (
            "num_transacoes_pgto_atrasado_mais_5_ult_30_dias",
            "num_transacoes_pgto_atrasado_mais_5_ult_60_dias",
            "razao_transacoes_pgto_atrasado_mais_5_30_60",
        ),
        (
            "num_transacoes_pgto_atrasado_mais_5_ult_30_dias",
            "num_transacoes_pgto_atrasado_mais_5_ult_90_dias",
            "razao_transacoes_pgto_atrasado_mais_5_30_90",
        ),
        (
            "num_transacoes_pgto_atrasado_mais_5_ult_30_dias",
            "num_transacoes_pgto_atrasado_mais_5_ult_120_dias",
            "razao_transacoes_pgto_atrasado_mais_5_30_120",
        ),
        (
            "num_transacoes_pgto_atrasado_menos_5_ult_30_dias",
            "num_transacoes_pgto_atrasado_menos_5_ult_60_dias",
            "razao_transacoes_pgto_atrasado_menos_5_30_60",
        ),
        (
            "num_transacoes_pgto_atrasado_menos_5_ult_30_dias",
            "num_transacoes_pgto_atrasado_menos_5_ult_90_dias",
            "razao_transacoes_pgto_atrasado_menos_5_30_90",
        ),
        (
            "num_transacoes_pgto_atrasado_menos_5_ult_30_dias",
            "num_transacoes_pgto_atrasado_menos_5_ult_120_dias",
            "razao_transacoes_pgto_atrasado_menos_5_30_120",
        ),
        (
            "num_transacoes_pgto_no_dia_ult_30_dias",
            "num_transacoes_pgto_no_dia_ult_60_dias",
            "razao_transacoes_pgto_no_dia_30_60",
        ),
        (
            "num_transacoes_pgto_no_dia_ult_30_dias",
            "num_transacoes_pgto_no_dia_ult_90_dias",
            "razao_transacoes_pgto_no_dia_30_90",
        ),
        (
            "num_transacoes_pgto_no_dia_ult_30_dias",
            "num_transacoes_pgto_no_dia_ult_120_dias",
            "razao_transacoes_pgto_no_dia_30_120",
        ),
        (
            "num_transacoes_pgto_adiantado_ult_30_dias",
            "num_transacoes_pgto_adiantado_ult_60_dias",
            "razao_transacoes_pgto_adiantado_30_60",
        ),
        (
            "num_transacoes_pgto_adiantado_ult_30_dias",
            "num_transacoes_pgto_adiantado_ult_90_dias",
            "razao_transacoes_pgto_adiantado_30_90",
        ),
        (
            "num_transacoes_pgto_adiantado_ult_30_dias",
            "num_transacoes_pgto_adiantado_ult_120_dias",
            "razao_transacoes_pgto_adiantado_30_120",
        ),
    ]

    # Calcula a razão para cada combinação de colunas
    for num_30, num_longer, razao_col in razao_cols:
        df[razao_col] = df[num_30] / df[num_longer].replace(0, fill_value)

    return df

def load_data():
    configure_oracle("8.3.0")
    engine = create_engine_from_string(config)
    folder = "data/inference"
    last_week_date = (datetime.now() - timedelta(days=7)).date()
    date_cols = [
        'Vencimento',
        'Data Pagamento', 
        'Inicio período', 
        'Fim período', 
        'data_consulta'
        ]

    output = f'{folder}/df_cobranca_infer.csv'

    if os.path.isfile(output):
        data = to_snake_case(pd.read_csv(output, parse_dates=date_cols))
    else:
        query = f'''
            SELECT * FROM "BOLEIA_SCHEMA"."V_COBRANCA" 
            WHERE "CNPJ Frota" IN 
            (SELECT "CNPJ Frota" FROM "BOLEIA_SCHEMA"."V_COBRANCA"
            WHERE "Fim período" >= TO_DATE('{last_week_date}','YYYY-MM-DD'))
            '''

        data = execute_query(engine, query, output)

    return data


def to_snake_case(df):
    # Usa a função underscore da biblioteca inflection para converter os nomes das colunas para snake_case
    df.columns = [unidecode(inflection.underscore(col).replace(' ','_')) for col in df.columns]
    return df

def create_features(df_base):
    base_cols = df_base.columns.to_list()
    df_last = df_base.sort_values(['cnpj_raiz','vencimento']).groupby('cnpj_raiz').last().reset_index()
    df_days = (df_base
        .pipe(add_days_columns)
        .pipe(transform_dataframe)
    )

    df_features = (df_base
        .pipe(create_num_cols, df_last)
        .pipe(add_prop_columns)
        .pipe(transform_dataframe)
        .merge(df_days, on=base_cols, how='left')
        .pipe(add_vigency_information)
        .pipe(add_razao_cols)
        .pipe(create_value_columns, df_last)
        .pipe(add_razao_valor_cols)
        .pipe(create_trans_columns, df_last)
        .pipe(add_razao_transacoes_cols)
    )

    return df_features


if __name__ == "__main__":

    ### Load Data
    df_cobranca = load_data()
    
    ### Feature Engineering
    df_inference = (df_cobranca
                    .pipe(preprocess_cobranca)
                    .pipe(transform_dataframe)
                    .pipe(create_features)
                   )
    df_inference.to_csv('data/inference/df_inference.csv', index=False)


