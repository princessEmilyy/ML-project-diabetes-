
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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
    lgb_params = {# "device_type": trial.suggest_categorical("device_type", ['gpu']),
                    #"num_estimators": trial.suggest_categorical("num_estimators", [1000,10000]), #10000
                    "num_estimators":  1000, #10000
                    "learning_rate": trial.suggest_float("learning_rate", 0.3, 0.4,step = 0.01),
                    "num_leaves": trial.suggest_int("num_leaves", 200, 270, step=5), #default 31 1000
                    #"max_depth": trial.suggest_int("max_depth", 3, 12), #defalt -1
                    "min_data_in_leaf":200,
                    #"min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 400, step=50),
                    "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=10),
                    "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=10),
                    "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.2, 0.3,step=0.001),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.9, step=0.1),
                    "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.75, 0.9, step=0.01),
                    "random_state": 42
                     }
    
    model = lgb.LGBMClassifier(**lgb_params)

    score_list = list()

    for fold in range(1,6):
        File = open(f'training_validation_clean_imputed_{fold}.pkl', 'rb')
        training_clean_imputed, val_clean_imputed = pickle.load(File)

        # Fit and transform the DataFrame
        # training_clean_imputed = pipeline.fit_transform(train)
        X_train = training_clean_imputed.drop('readmitted', axis=1)
        y_train = training_clean_imputed['readmitted']
        y_train = LabelEncoder().fit_transform(y_train)
        # val = pd.read_csv(f'validation_fold_{fold}.csv')
        # val = val[list(set(val.columns).intersection(set(train.columns)))]
        # val_clean_imputed = pipeline.fit_transform(val)
        X_val = val_clean_imputed.drop('readmitted', axis=1)
        y_val = val_clean_imputed['readmitted']
        y_val = LabelEncoder().fit_transform(y_val)
        cl_list = list(set(X_val.columns).intersection(set(X_train.columns))) #QC
        X_train = X_train[cl_list] #same order
        X_val = X_val[cl_list]
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_pred = model.predict(X_val)
        score_list.append(balanced_accuracy_score(y_val, y_pred))
        #File = open(f'training_validation_clean_imputed_{fold}.pkl', 'wb')
        #pickle.dump((training_clean_imputed, val_clean_imputed), File)
    return sum(score_list) / len(score_list),  #Mean balanced accuracy

# Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, timeout=400)
print('Best hyperparameters:', study.best_params)
print('Best balance accuracy:', study.best_value)

f = open("LGBM_hyperparameters_.txt","a") #"w" check for append
f.write(f'Best params:{study.best_params}/n/n'
        f'Mean CV score (balanced_accuracy)::{study.best_value}/n/n')
f.close

### Visualizing Optuna results ###

#Plot optimization history of all trials in a study
# fig_optuna = optuna.visualization.plot_optimization_history(study)
# fig_optuna.show()

# fig_importance = optuna.visualization.plot_param_importances(
#     study, target=lambda t: t.duration.total_seconds(), target_name="duration")
# fig_importance.show()