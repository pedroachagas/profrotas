import numpy as np
import pandas as pd
import pickle
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import lightgbm as lgb
from xgboost import XGBClassifier
from tqdm import tqdm

# Função para calcular a área sob a curva ROC e a estatística KS
def ks_statistic(y_true, y_pred):
    # Calcula o KS-Statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    return max(tpr - fpr)

def score_model(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    ks = ks_statistic(y_true, y_pred)
    aucpr = average_precision_score(y_true, y_pred)
    return auc, ks, aucpr


def train_xgboost_model(parameters, x, y, cat_cols=None):

    if cat_cols:

        x = x.copy()
        for col in cat_cols:
            x[col] = x[col].astype('category')

        model = XGBClassifier(**parameters, verbosity=0, categorical_features=cat_cols, enable_categorical=True, tree_method='gpu_hist')
        model.fit(x, y)
    else:
        model = XGBClassifier(**parameters, verbosity=0, tree_method='gpu_hist')
        model.fit(x, y)
        
    return model

def train_lightgbm_model(parameters, x, y, cat_features):
    categorical_indices = [i for i, col in enumerate(x.columns) if col in cat_features]
    
    x = x.copy()
    for col in cat_features:
        x[col] = x[col].astype('category')
    
    if len(categorical_indices) > 0:
        model = lgb.LGBMClassifier(**parameters)
        model.fit(x, y, categorical_feature=cat_features)
    else:
        model = lgb.LGBMClassifier(**parameters)
        model.fit(x, y)

    return model

def train_model(model_name, parameters, x, y, cat_cols=None):
    if model_name == 'xgboost':
        return train_xgboost_model(parameters, x, y, cat_cols)
    elif model_name == 'lightgbm':
        return train_lightgbm_model(parameters, x, y, cat_cols)
    else:
        raise ValueError('Modelo inválido')

def predict(model, x, cat_cols=None):
       
    if model.__class__.__name__ == 'LGBMClassifier' and cat_cols is not None:
        x = x.copy()
        for col in cat_cols:
            x[col] = x[col].astype('category')

    if model.__class__.__name__ == 'XGBClassifier' and cat_cols is not None:
        x = x.copy()
        for col in cat_cols:
            x[col] = x[col].astype('category')

    return model.predict_proba(x)[:, 1]

def process_dataframe(df):
    df = df.copy()
    df.target = df.target.apply(lambda x: 
                   'em_dia' if x in ['pgto_adiantado','pgto_no_dia','pgto_atrasado_menos_5'] else
                   'atrasado' if x == 'pgto_atrasado_mais_5' else
                   'vencido'
                   )
    df.sort_values('vencimento', inplace=True)
    return df


def split_dataframe(df):
    cutoff_row = int(len(df) * 0.75)
    cutoff_date = df.iloc[cutoff_row].vencimento
    df_treino = df.loc[df['vencimento'] < cutoff_date,:]
    df_oot = df.loc[df['vencimento'] >= cutoff_date]
    report_datasets(df_treino, df_oot)
    save_datasets(df_treino, df_oot)
    return df_treino, df_oot


def save_datasets(df_treino, df_oot):
    df_treino.to_csv('data/training/df_treino.csv', index=False)
    df_oot.to_csv('data/training/df_oot.csv', index=False)


def report_datasets(df_treino, df_oot):
    print('\n--- Treino ---')
    print('\nNúm. linhas:', len(df_treino), f'({len(df_treino) / len(df_ml_complete) * 100:.2f}%)')
    print('\nTarget:')
    print(df_treino.target.value_counts())
    print('\n')

    print('--- OOT ---')
    print('\nNúm. linhas:', len(df_oot), f'({len(df_oot) / len(df_ml_complete) * 100:.2f}%)')
    print('\nTarget:')
    print(df_oot.target.value_counts())


def create_training_data(df_treino, df_oot):
    report_datasets(df_treino, df_oot)
    X_train = df_treino.drop('target', axis=1).set_index(['cnpj_raiz','vencimento']).fillna(0)
    y_train = df_treino.target.apply(lambda x: 1 if x == ['vencido','atrasado'] else 0)
    X_oot = df_oot.drop('target', axis=1).set_index(['cnpj_raiz','vencimento']).fillna(0)
    y_oot = df_oot.target.apply(lambda x: 1 if x in ['vencido','atrasado'] else 0)

    binary_cols = [col for col in X_train.columns if set(X_train[col].unique()) == {0, 1}]
    cat_features = list(set(X_train.select_dtypes(include=['object', 'category']).columns.tolist() + binary_cols))

    return X_train, y_train, X_oot, y_oot, cat_features

if __name__ == '__main__':
    df_ml_complete = pd.read_csv('data/processed/df_ml.csv', parse_dates=['vencimento'])
    df_ml_complete = process_dataframe(df_ml_complete)
    
    df_treino, df_oot = split_dataframe(df_ml_complete)
    X_train, y_train, X_oot, y_oot, cat_features = create_training_data(df_treino, df_oot)

    selected_features = pickle.load(open('data/optimization/features_selecionadas.pkl', 'rb'))
    best_params = pickle.load(open('data/optimization/melhores_parametros.pkl', 'rb'))
    results = pd.read_csv('data/optimization/resultados_experimentos.csv')

    best_model = results.sort_values('auc_oot', ascending=False).iloc[0]
    model_name = best_model['model_name']
    feature_set = best_model['feature_set']
    print(f'\nMelhor modelo: {model_name} - {feature_set}')

    params = best_params[f'{model_name}_{feature_set}']
    features = selected_features[f'{feature_set}']

    X_oot_selected = X_oot[features].copy()
    X_train_selected = X_train[features].copy()

    cat_cols = [c for c in features if c in cat_features]

    model = train_model(model_name, params, X_train_selected, y_train, cat_cols)

    y_pred = predict(model, X_oot_selected, cat_cols)
    print('Contagem de cada tipo de valores preditos:')
    print(pd.Series(y_pred).value_counts())

    auc, ks, auc_pr = score_model(y_oot, y_pred)
    print('AUC: {:.4f}'.format(auc))
    print('KS: {:.4f}'.format(ks))
    print('AUC PR: {:.4f}'.format(auc_pr))

    pickle.dump(model, open(f'data/training/model_{model_name}_{feature_set}.pkl', 'wb'))

