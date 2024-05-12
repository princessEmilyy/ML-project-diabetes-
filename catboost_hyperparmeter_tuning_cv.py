import catboost as cb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
import optuna
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import Class_ML_Project
from sklearn.pipeline import Pipeline



CATEGORICAL = ['race', 'gender', 'medical_specialty', 'max_glu_serum',
               'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
               'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
               'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
               'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
               'diabetesMed', 'admission_type_descriptor', 'discharge_disposition_descriptor',
               'admission_source_descriptor']

# Define numerical feature list
NUMERICAL = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
             'number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient','age']

# Define irrelevant feature list
IRRELEVANT_FEATURES = [ 'nateglinide','chlorpropamide','tolbutamide','acarbose','miglitol','troglitazone',
 'tolazamide','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-pioglitazone',
 'admission_source_descriptor','repaglinide']


# Define columns for OHE
OHE_regular_cols = ['race', 'gender', 'medical_specialty', 'insulin', 'diabetesMed', 'admission_type_descriptor',
                    'discharge_disposition_descriptor']
OHE_4_to_2_cols = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
diagnoses_cols = ['diag_1_cat', 'diag_2_cat', 'diag_3_cat']



def objective(trial):
    #{ 'random_strength': 1.2300217338005526, 'depth': 6, 'min_data_in_leaf': 299, 'n_estimators': 857}
    cb_params = {
        # 'iterations': 1000,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.1, 0.16),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 87, 90),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.9, 1.1),
        'random_strength': trial.suggest_float('random_strength', 1, 1.4),
        'depth': trial.suggest_int('depth', 5, 7),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 800, 900),
        # "use_best_model": True,
        # "task_type": "GPU",
        # 'devices': '0:1:3',
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'random_seed': 42
    }

    #num_boost_round = params_search.pop('num_estimators')

    #cls_cv = cb.cv(params=params_search, num_boost_round=num_boost_round, dtrain=dtrain, nfold=nfold, shuffle=True,
    #                metrics=eval_metric, stratified=True, seed=42, folds=folds_idx)

    model = cb.CatBoostClassifier(**cb_params)
    # CatBoostClassifier(n_estimators=3, random_state=42)
    score_list = list()
    for fold in range(1,6):
        train_real = pd.read_csv(f"train_fold_{fold}.csv")
        train_GAN = pd.read_csv(f"CTGAN_{fold}_new.csv")
        train_real = train_real[ list(set(train_GAN.columns).intersection(set(train_real.columns)))]
        train = pd.concat([train_real,train_GAN])
        pipeline = Pipeline(
            [('feature_remover', Class_ML_Project.FeatureRemover(features_to_remove=IRRELEVANT_FEATURES)),
             ('imputer_race',
              Class_ML_Project.DataFrameImputer(strategy='constant', fill_value='other', columns=['race'])),
             ('imputer_medical',
              Class_ML_Project.DataFrameImputer(strategy='most_frequent', columns=['medical_specialty'])),
             ('age_encoder', Class_ML_Project.MultiColumnLabelEncoder(columns=['age'])),
             ('numerical_scaler', Class_ML_Project.NumericalTransformer(columns=NUMERICAL)),
             (
             'OHE', Class_ML_Project.CustomOHEncoder(OHE_regular_cols=OHE_regular_cols, OHE_4_to_2_cols=OHE_4_to_2_cols,
                                                     change_col='change', diag_cols=diagnoses_cols))])

        # Fit and transform the DataFrame
        training_clean_imputed = pipeline.fit_transform(train)
        X_train = training_clean_imputed.drop('readmitted', axis=1)
        y_train = training_clean_imputed['readmitted']
        y_train = LabelEncoder().fit_transform(y_train)
        val = pd.read_csv(f'validation_fold_{fold}.csv')
        val = val[list(set(val.columns).intersection(set(train.columns)))]
        val_clean_imputed = pipeline.fit_transform(val)
        X_val = val_clean_imputed.drop('readmitted', axis=1)
        y_val = val_clean_imputed['readmitted']
        y_val = LabelEncoder().fit_transform(y_val)
        cl_list = list(set(X_val.columns).intersection(set(X_train.columns)))
        X_train = X_train[cl_list]
        X_val = X_val[cl_list]
        if 'Unnamed: 0' in X_train.columns:
            print()
        elif 'Unnamed: 0' in X_val.columns:
            print()
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        y_pred = model.predict(X_val)
        score_list.append(balanced_accuracy_score(y_val, y_pred))
    return sum(score_list) / len(score_list)

File = open('Training_data-tuple_x_y.pkl', 'rb')
X, y = pickle.load(File)


# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for train_idx, val_idx in kf.split(X):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, timeout=400)
print('Best hyperparameters:', study.best_params)
print('Best balance accuracy:', study.best_value)
