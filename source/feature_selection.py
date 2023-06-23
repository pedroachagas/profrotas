import pandas as pd
from catboost import CatBoostClassifier, Pool, EFeaturesSelectionAlgorithm, EShapCalcType
import pickle

def process_dataframe(df):
    df.target = df.target.apply(lambda x: 
                   'em_dia' if x in ['pgto_adiantado','pgto_no_dia','pgto_atrasado_menos_5'] else
                   'atrasado' if x == 'pgto_atrasado_mais_5' else
                   'vencido'
                   )
    df = df.sort_values('vencimento')
    return df


def select_features(df_treino):
    X = df_treino.drop('target', axis=1).set_index(['cnpj_raiz','vencimento']).fillna(0)
    y = df_treino.target.apply(lambda x: 1 if x == 'vencido' else 0)

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].astype('category')

    return X, y, cat_features


def catboost_feature_selection(X, y, cat_features):
    model = CatBoostClassifier(iterations=600, random_seed=0)
    X_features = X.columns.tolist()
    train_pool = Pool(X, y, feature_names=X_features, cat_features=cat_features)
    
    selected_features = {}
    for n in [10, 15, 20, 30]:
        summary = model.select_features(
            train_pool,
            features_for_select=X_features,
            num_features_to_select=n,
            steps=50,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=False,
            logging_level="Silent",
        )
        selected_features[f'top_{n}'] = summary['selected_features_names']
        print(f'Selected top {n} features')
    
    return selected_features


def save_to_file(selected_features):
    pickle.dump(selected_features, open('data/optimization/features_selecionadas.pkl', 'wb'))

if __name__ == "__main__":
    df_ml_complete = pd.read_csv('data/optimization/df_ml.csv', parse_dates=['vencimento'])
    df_ml_complete = process_dataframe(df_ml_complete)
    X, y, cat_features = select_features(df_ml_complete)
    selected_features = catboost_feature_selection(X, y, cat_features)
    save_to_file(selected_features)
