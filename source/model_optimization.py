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
    fpr, tpr, _ = roc_curve(y_true, y_pred)

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

  
def get_metric(y, y_pred):
    return roc_auc_score(y, y_pred)


def decision(metric_train, metric_test, thr=0.05):
    
    return 0 if np.abs(metric_train - metric_test) > thr else metric_test


def generate_objective_function_xgboost(x, y, cat_features):

    def objective(trial, x=x, y=y, cat_features=cat_features):
        
        parameters = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 400, 500, 600]),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "eta": trial.suggest_float("eta", 0.001, 0.3, log=True),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
        } 
        train_metrics = np.zeros(5)
        test_metrics = np.zeros(5)

        kf = StratifiedKFold(shuffle=True, random_state=0)
        
        for i, (idx_train, idx_test) in enumerate(kf.split(x, y)):

            x_train, y_train = x.iloc[idx_train], y.iloc[idx_train]
            x_test, y_test = x.iloc[idx_test], y.iloc[idx_test]
            
            model = train_xgboost_model(parameters, x_train, y_train, cat_features)

            y_pred_train = predict(model, x_train, cat_features)
            y_pred_test = predict(model, x_test, cat_features)

            train_metrics[i] = get_metric(y_train, y_pred_train)
            test_metrics[i] = get_metric(y_test, y_pred_test)
            
        metrics = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_metric": np.mean(train_metrics),
            "test_metric": np.mean(test_metrics),
            "test_metric": np.mean(test_metrics),
            "train_std": np.std(train_metrics),
            "test_std": np.std(test_metrics),   
        }
        
        for key in metrics:
            
            trial.set_user_attr(key, metrics[key])
        
        return decision(np.mean(train_metrics), np.mean(test_metrics))
    
    return objective


def generate_objective_function_lgbm(x, y, cat_features):
    def objective(trial, x=x, y=y, cat_features=cat_features):
        parameters = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0, step=0.1),  # Changed to subsample
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),  # Changed to subsample_freq
            "cat_smooth": trial.suggest_float("cat_smooth", 1e-2, 1, step=1e-2),
        }

        train_metrics = np.zeros(5)
        test_metrics = np.zeros(5)

        kf = StratifiedKFold(shuffle=True, random_state=0)

        for i, (idx_train, idx_test) in enumerate(kf.split(x, y)):
            x_train, y_train = x.iloc[idx_train], y.iloc[idx_train]
            x_test, y_test = x.iloc[idx_test], y.iloc[idx_test]

            model = train_lightgbm_model(parameters, x_train, y_train, cat_features)

            y_pred_train = predict(model, x_train,cat_features)
            y_pred_test = predict(model, x_test,cat_features)

            train_metrics[i] = get_metric(y_train, y_pred_train)
            test_metrics[i] = get_metric(y_test, y_pred_test)

        metrics = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_metric": np.mean(train_metrics),
            "test_metric": np.mean(test_metrics),
            "train_std": np.std(train_metrics),
            "test_std": np.std(test_metrics),
        }

        for key in metrics:
            trial.set_user_attr(key, metrics[key])

        return decision(np.mean(train_metrics), np.mean(test_metrics))

    return objective


def create_optuna_callback(pbar):
    def optuna_callback(study, trial):

        pbar.update(1)

    return optuna_callback


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
    report_datasets(df_treino, df_oot, df)
    save_datasets(df_treino, df_oot)
    return df_treino, df_oot


def save_datasets(df_treino, df_oot):
    df_treino.to_csv('../data/training/df_treino.csv', index=False)
    df_oot.to_csv('../data/training/df_oot.csv', index=False)


def report_datasets(df_treino, df_oot, df_ml_complete):
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
    X = df_treino.drop('target', axis=1).set_index(['cnpj_raiz','vencimento']).fillna(0)
    y = df_treino.target.apply(lambda x: 1 if x == ['vencido','atrasado'] else 0)
    X_oot = df_oot.drop('target', axis=1).set_index(['cnpj_raiz','vencimento']).fillna(0)
    y_oot = df_oot.target.apply(lambda x: 1 if x in ['vencido','atrasado'] else 0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)

    binary_cols = [col for col in X.columns if set(X[col].unique()) == {0, 1}]
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist() + binary_cols
    
    for col in cat_features:
        X[col] = X[col].astype('category')

    return X_train, y_train, X_val, y_val, X_oot, y_oot, cat_features


def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, cat_features, selected_features):

    model_funcs = {
        'xgboost': [generate_objective_function_xgboost, train_xgboost_model],
        'lightgbm': [generate_objective_function_lgbm, train_lightgbm_model],
    }

    n_trials = 60

    best_models = {}
    best_params = {}
    results_df = pd.DataFrame(columns=['model_name', 'feature_set', 'auc_val', 'ks_val', 'aucpr_val', 'auc_oot', 'ks_oot', 'aucpr_oot'])
    n_rounds = len(model_funcs.keys()) * len(selected_features.keys()) * n_trials

    with tqdm(total=n_rounds, desc='Modeling Experiments') as pbar:

        optuna_callback = create_optuna_callback(pbar)
        
        for n, features in selected_features.items():
            
            X_train_selected = X_train[features].copy()
            X_val_selected = X_val[features].copy()
            X_oot_selected = X_oot[features].copy()
            
            for model, funcs in model_funcs.items():

                gererate_objective_function, train_model = funcs
                    
                cat_cols = [c for c in features if c in cat_features]

                objective = gererate_objective_function(X_train_selected, y_train, cat_cols)

                optuna.logging.set_verbosity(optuna.logging.WARNING)

                study = optuna.create_study(directions=["maximize"])

                study.optimize(objective, n_trials=n_trials, gc_after_trial=True, callbacks=[optuna_callback])

                best_model = train_model(study.best_params, X_train_selected, y_train, cat_cols)

                best_models[model + '_' + n] = best_model
                best_params[model + '_' + n] = study.best_params
                
                # Validation set
                y_pred_prob_val = predict(best_model, X_val_selected, cat_cols)
                auc_val, ks_val, aucpr_val = score_model(y_val, y_pred_prob_val)

                # OOT set
                y_pred_prob_oot = predict(best_model, X_oot_selected, cat_cols)
                auc_oot, ks_oot, aucpr_oot = score_model(y_oot, y_pred_prob_oot)

                # Create a new dataframe with the results
                new_results = pd.DataFrame({
                    'model_name': [model],
                    'feature_set': [n],
                    'auc_val': [auc_val],
                    'ks_val': [ks_val],
                    'aucpr_val': [aucpr_val],
                    'auc_oot': [auc_oot],
                    'ks_oot': [ks_oot],
                    'aucpr_oot': [aucpr_oot]
                })

                # Concatenate the new results to the existing dataframe
                results_df = pd.concat([results_df, new_results], ignore_index=True)
            
                print(f'Modeling for {model} with {n.replace("_"," ")} completed!')

    results_df.to_csv('../data/training/resultados_experimentos.csv', index=False)
    pickle.dump(best_models, open('data/optimization/melhores_modelos.pkl', 'wb'))
    pickle.dump(best_params, open('data/optimization/melhores_parametros.pkl', 'wb'))
    

if __name__ == '__main__':
    df_ml_complete = pd.read_csv('data/processed/df_ml.csv', parse_dates=['vencimento'])
    df_ml_complete = process_dataframe(df_ml_complete)
    
    df_treino, df_oot = split_dataframe(df_ml_complete)
    X_train, y_train, X_val, y_val, X_oot, y_oot, cat_features = create_training_data(df_treino, df_oot)

    selected_features = pickle.load(open('data/optimization/features_selecionadas.pkl', 'rb'))
    train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, cat_features, selected_features)