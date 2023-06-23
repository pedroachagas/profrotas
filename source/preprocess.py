import pandas as pd
import numpy as np

import inflection
from unidecode import unidecode

import os

from datetime import datetime as dt
from workadays import workdays as wd

## Funções Auxiliares

# Função que converte os nomes das colunas de um DataFrame para snake_case
def to_snake_case(df):
    # Usa a função underscore da biblioteca inflection para converter os nomes das colunas para snake_case
    df.columns = [unidecode(inflection.underscore(col).replace(' ','_')) for col in df.columns]
    return df

def format_cnpj(input_string):
    input_string = list(input_string)  # Convert string to list because strings are immutable in Python

    if len(input_string) != 14:
        return 'Input string must have exactly 14 characters'
    
    input_string.insert(2, '.')
    input_string.insert(6, '.')
    input_string.insert(10, '/')
    input_string.insert(15, '-')
    
    return ''.join(input_string)  # Convert list back to string

def load_data(cobranca_path, frota_path):
    print("Carregando dados...")
    df_cobranca = to_snake_case(pd.read_csv(cobranca_path, parse_dates=['Vencimento', 'Data Pagamento', 'Inicio período', 'Fim período']))
    print("Base COBRANCAS carregada.")
    df_frota = to_snake_case(pd.read_csv(frota_path, parse_dates=['Data Inicio Contrato']))
    print("Base FROTAS carregada.")
    return df_cobranca, df_frota


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


## Funções de pré-processamento

### Base de cobrança

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
    cond4 = df_cobranca.data_atraso < df_cobranca.data_consulta
    condicoes = cond1 & cond2 & cond3 & cond4
    df_cobranca_filtered = (df_cobranca[condicoes].pipe(filter_unpaid_after_first).pipe(remove_same_day_duplicates))
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
    return df_processed


### Base de contas

def preprocess_dates_frotas(df):
    df.data_inicio_contrato = pd.to_datetime(df.data_inicio_contrato.dt.date)
    df.data_consulta = pd.to_datetime(df.data_consulta)
    return df


def preprocess_cargo(title):
    title = unidecode(title).lower()
    cargo_map = [
        ("proprietario", "president"),
        ("proprietario", "prop"),
        ("proprietario", "dono"),
        ("proprietario", "dona"),
        ("socio", "soc"),
        ("socio", "ceo"),
        ("diretor", "dir"),
        ("diretor", "execut"),
        ("gerente", "ger"),
        ("gerente", "ges"),
        ("gerente", "sup"),
        ("gerente", "coord"),
        ("gerente", "adm"),
    ]

    for new, old in cargo_map:
        if old in title:
            return new
    return np.nan if title == "nan" else "outros"


def preprocess_estado(estado):
    outros_list = [
        "RO",
        "MT",
        "PE",
        "AM",
        "MA",
        "MS",
        "DF",
        "TO",
        "CE",
        "AC",
        "PA",
        "RN",
        "PI",
        "RR",
        "SE",
        "AP",
        "AL"
    ]
    if estado in outros_list:
        return "Outros"
    else:
        return estado
    

def is_capital(city):
    capitais = [
        "Distrito Federal",
        "Rio Branco",
        "Maceió",
        "Macapá",
        "Manaus",
        "Salvador",
        "Fortaleza",
        "Brasília",
        "Vitória",
        "Goiânia",
        "São Luís",
        "Cuiabá",
        "Campo Grande",
        "Belo Horizonte",
        "Belém",
        "João Pessoa",
        "Curitiba",
        "Recife",
        "Teresina",
        "Rio de Janeiro",
        "Natal",
        "Porto Alegre",
        "Porto Velho",
        "Boa Vista",
        "Florianópolis",
        "São Paulo",
        "Aracaju",
    ]
    capitais = sorted([unidecode(capital).lower() for capital in capitais])
    return int(city.lower() in capitais)


def preprocess_email(email):
    public_domain_substrings = [
        "gmail",
        "yahoo",
        "outlook",
        "hotmail",
        "live",
        "bol",
        "gmai",
        "globo",
        "msn",
        "icloud",
    ]   
    domain = email.split("@")[-1].split(".")[0]
    return int(domain not in public_domain_substrings)


def create_cols_df_frotas(df):
    def mode(series):
        modes = series.mode()
        return modes.iloc[0] if not modes.empty else None

    df = df[df.modalidade == 1].copy()

    # Create cols
    df['cargo_dono_agg'] = df.cargo_dono.apply(preprocess_cargo)
    df['estado'] = df.estado.apply(preprocess_estado)
    df['email_corporativo'] = df.email_dono.apply(preprocess_email)
    df["capital"] = df.municipio.apply(is_capital)

    # Aggregate df by cnpj_raiz
    df_features = (
        df[
            [
                "cnpj_raiz",
                "cargo_dono_agg",
                "estado",
                "email_corporativo",
                "capital",
            ]
        ]
        .groupby(["cnpj_raiz"])
        .agg(
            {
                "email_corporativo": mode,
                "cargo_dono_agg": mode,
                "estado": mode,
                "capital": mode,
            }
        )
    )

    return df_features.reset_index()

def preprocess_frotas(df):
    print("Começando processamento da base FROTAS...")
    df_processed = (
            df.pipe(preprocess_cnpj)
                .pipe(preprocess_dates_frotas)
                .pipe(create_cols_df_frotas)
    )
    print("Processamento concluído.")
    return df_processed

## Main script

if __name__ == "__main__":

    cobranca_path = f'data/df_cobranca.csv'
    frota_path = f'data/df_frota.csv'

    df_cobranca, df_frota = load_data(cobranca_path, frota_path)
    df_cobranca_processed = preprocess_cobranca(df_cobranca)
    df_cobranca_processed.to_csv(f'data/processed/df_cobranca_processed.csv', index=False)
    df_frota_processed = preprocess_frotas(df_frota)
    df_frota_processed.to_csv(f'data/processed/df_frota_processed.csv', index=False)
    print("Bases processadas com sucesso.")