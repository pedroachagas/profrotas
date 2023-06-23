import pandas as pd
import numpy as np

from tqdm import tqdm

import os
import shutil

from sklearn.linear_model import LinearRegression


# Função para criar novas colunas com o número de linhas filtradas
def create_num_cols(df):
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

    total_entries = len(df.index)
    progress_bar = tqdm(desc='Criando features de contagem de cobranças',total=total_entries * len(targets) * (len(intervalos_de_tempo) + 1))

    # Cria uma nova coluna para cada target e para cada intervalo de tempo
    df["num_cobrancas"] = 0
    for i in df.index:
        # Usa a função filtered_df para obter o número de linhas filtradas
        df.loc[i, "num_cobrancas"] = filtered_df(
            i[0], df.loc[i, "data_lancamento"], None, None, df
        )
    for target in targets:
        total_col = f"num_{target}_total"
        df[total_col] = 0
        for i in df.index:
            # Usa a função filtered_df para obter o número de linhas filtradas
            df.loc[i, total_col] = filtered_df(
                i[0], df.loc[i, "data_lancamento"], None, target, df
            )
            
            progress_bar.update(1)
            
        for periodo in intervalos_de_tempo:
            col_name = f"num_{target}_ult_{periodo}_dias"
            df[col_name] = 0
            for i in df.index:
                # Usa a função filtered_df para obter o número de linhas filtradas
                df.loc[i, col_name] = filtered_df(
                    i[0], df.loc[i, "data_lancamento"], periodo, target, df
                )

                progress_bar.update(1)

    progress_bar.close()

    # Reorganiza o índice e remove a coluna "index"
    return df.sort_index().reset_index().drop("index", axis=1)


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
    for num_30, num_longer, razao_col in razao_cols:
        df[razao_col] = df[num_30] / df[num_longer].replace(0, fill_value)

    return df

# Função para criar novas colunas com o valor total das linhas filtradas
def create_value_columns(df):
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
    df.reset_index(inplace=True)
    df.set_index(["cnpj_raiz", "vencimento"], inplace=True)
    df.sort_index(inplace=True)

    total_entries = len(df.index)
    progress_bar = tqdm(desc='Criando features de valores das cobranças',total=total_entries * len(targets) * (len(intervalos_de_tempo) + 1))

    # Cria uma nova coluna para cada target e para cada intervalo de tempo
    for target in targets:
        total_col = f"valor_{target}_total"
        df[total_col] = 0
        for i in df.index:
            # Usa a função filtered_value para obter o número de linhas filtradas
            df.loc[i, total_col] = filtered_value(
                i[0], df.loc[i, "data_lancamento"], None, target, df
            )

            progress_bar.update(1)

        for periodo in intervalos_de_tempo:
            col_name = f"valor_{target}_ult_{periodo}_dias"
            df[col_name] = 0
            for i in df.index:
                # Usa a função filtered_value para obter o número de linhas filtradas
                df.loc[i, col_name] = filtered_value(
                    i[0], df.loc[i, "data_lancamento"], periodo, target, df
                )

                progress_bar.update(1)

    progress_bar.close()

    # Reorganiza o índice e remove a coluna "index"
    return df.sort_index().reset_index().drop("index", axis=1)

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
def create_trans_columns(df):
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

    total_entries = len(df.index)
    progress_bar = tqdm(desc='Criando features de número de transações',total=total_entries * len(targets) * (len(intervalos_de_tempo) + 1))

    # Cria uma nova coluna para cada target e para cada intervalo de tempo
    for target in targets:
        total_col = f"num_transacoes_{target}_total"
        df[total_col] = 0
        for i in df.index:
            # Usa a função filtered_df para obter o número de linhas filtradas
            df.loc[i, total_col] = filtered_trans(
                i[0], df.loc[i, "data_lancamento"], None, target, df
            )

            progress_bar.update(1)
        for periodo in intervalos_de_tempo:
            col_name = f"num_transacoes_{target}_ult_{periodo}_dias"
            df[col_name] = 0
            for i in df.index:
                # Usa a função filtered_df para obter o número de linhas filtradas
                df.loc[i, col_name] = filtered_trans(
                    i[0], df.loc[i, "data_lancamento"], periodo, target, df
                )

                progress_bar.update(1)

    progress_bar.close()

    # Reorganiza o índice e remove a coluna "index"
    return df.sort_index().reset_index().drop("index", axis=1)


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


def create_features(df):
    # Define the create_features checkpoints folder
    checkpoints_folder = "data/checkpoints"
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Define checkpoint file paths
    num_cols_checkpoint = os.path.join(checkpoints_folder, "num_cols.csv")
    days_columns_checkpoint = os.path.join(checkpoints_folder, "days_columns.csv")
    value_columns_checkpoint = os.path.join(checkpoints_folder, "value_columns.csv")
    trans_columns_checkpoint = os.path.join(checkpoints_folder, "trans_columns.csv")

    # Contagem, proporção e razão de cobranças
    if not os.path.exists(num_cols_checkpoint):
        print('Criando colunas de contagem, proporção e razão de cobranças...')
        df = create_num_cols(df)
        df = add_prop_columns(df)
        df = add_razao_cols(df)
        df.to_csv(num_cols_checkpoint, index=False)
        print('Colunas de contagem, proporção e razão de cobranças criadas com sucesso.')
    else:
        df = pd.read_csv(num_cols_checkpoint)

    # Dias de atraso e vigência
    if not os.path.exists(days_columns_checkpoint):
        df = add_days_columns(df)
        df = add_vigency_information(df)
        df.to_csv(days_columns_checkpoint, index=False)
        print('Colunas de dias de atraso e vigência criadas com sucesso.')
    else:
        df = pd.read_csv(days_columns_checkpoint)

    # Soma e razão de valores totais
    if not os.path.exists(value_columns_checkpoint):
        df = create_value_columns(df)
        df = add_razao_valor_cols(df)
        df.to_csv(value_columns_checkpoint, index=False)
        print('Colunas de soma e razão de valores totais criadas com sucesso.')
    else:
        df = pd.read_csv(value_columns_checkpoint)

    # Soma e razão do número de transações
    if not os.path.exists(trans_columns_checkpoint):
        df = create_trans_columns(df)
        df = add_razao_transacoes_cols(df)
        df.to_csv(trans_columns_checkpoint, index=False)
        print('Colunas de soma e razão do número de transações criadas com sucesso.')
    else:
        df = pd.read_csv(trans_columns_checkpoint)

    print('Features criadas com sucesso.')
    
    # Delete the checkpoints folder
    shutil.rmtree(checkpoints_folder)

    return df


def filter_and_create_ml_df(df, days):
    # Filter the DataFrame to keep only rows with dias_vigencia > days
    df = df[df.dias_vigencia > days]

    # Remove unnecessary columns from the DataFrame, reset the index, and make a copy
    df_ml = df.drop(
        columns=[
            "data_pagamento",
            "data_atraso",
            "data_lancamento",
            "transacoes",
            "valor_total",
            "dias_atraso",
        ]
    )

    # Return the resulting DataFrame
    return df_ml

if __name__ == "__main__":

    # Carregar os dados
    print('Carregando os dados...')
    df_base = pd.read_csv('data/processed/df_cobranca_processed.csv', parse_dates=['vencimento','data_pagamento','data_atraso','data_lancamento'])
    df_frota = pd.read_csv('data/processed/df_frota_processed.csv')
    print('Dados carregados com sucesso.')

    # Criar as features
    print('Criando as features da base COBRANCA...')
    df_ml = (df_base
        # .sample(frac=0.1, random_state=42) ## Teste com uma amostra menor
        .pipe(create_features)
        .pipe(filter_and_create_ml_df, days=60)
    )
    print('Features criadas com sucesso.')
    print('Unindo com a base de frota...')
    df_ml_complete = df_ml.merge(df_frota, on="cnpj_raiz", how="left")
    print('Base de cobrança unida com a base de frota com sucesso.')

    # Salvar a base de cobrança
    print('Salvando a base de cobrança...')
    df_ml_complete.to_csv('data/processed/df_ml.csv', index=False)