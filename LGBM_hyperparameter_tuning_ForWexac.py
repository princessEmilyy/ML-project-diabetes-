##########################    LGBM hyperparameter-tuning     ##########################
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import randint, uniform
import pickle
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score,train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFoldprecision_score
from sklearn.metrics import make_scorer,accuracy_score,balanced_accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
import optuna
import time

# Read the training set
file_path = "after_pipeline_dataset.csv"
train_data = pd.read_csv(file_path,index_col=0)
#print(train_data.head())

# for column in train_data.columns:
#     print(f"{column}: {train_data[column].dtype}")

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Fit label encoder and transform the target column
train_data['readmitted'] = label_encoder.fit_transform(train_data['readmitted'])

# Define X and y
X = train_data.drop(columns=['readmitted'])
y = train_data['readmitted']

def objective(trial):
    lgbm_params = {
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50), 
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0), 
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.0, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 777,
        'device': 'gpu',
        'cuda': True
    }
    
    # Perform cross-validation with early stopping
    cv_results = lgb.cv(lgbm_params, lgb.Dataset(X, y), num_boost_round=1000, nfold=5, metrics='balanced_accuracy_score', seed=42)
    
    # Check if 'balanced_accuracy_score' is present in cv_results
    if 'balanced_accuracy_score' in cv_results:
        # Compute mean precision across folds
        mean_balanced_accuracy = max(cv_results['balanced_accuracy_score'])  # Assuming maximizing precision
        return mean_balanced_accuracy
    else:
        # If 'balanced_accuracy_score' is not present or computation fails, return a very low value
        return -float('inf')

# Define the timeout handler
def timeout_handler(study, trial):
    # This function will be called when the timeout is reached
    raise optuna.TrialPruned()

# Start tracking time
start_time = time.time()

# Perform hyperparameter optimization with Optuna
study = optuna.create_study(direction='maximize',timeout=600)  # Note: 'maximize' since we're maximizing precision
study.optimize(objective, n_trials=100,timeout_handler=timeout_handler)

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Optimization took {elapsed_time:.2f} seconds")

# Get best parameters if any trial is successful
if len(study.trials) > 0:
    best_params = study.best_params
    print('Best params:', best_params)

    # Perform cross-validation with best parameters
    best_cv_score = study.best_value  # Optuna maximizes, so we directly use the best value for precision
    precision_scorer = 'precision'  # No need for make_scorer as precision is directly supported by sklearn
    cross_val_scores = cross_val_score(lgb.LGBMClassifier(**best_params), X, y, cv=5, scoring=precision_scorer)

    # Print cross-validation results
    print('Cross-validation results:')
    print('Best CV score (precision):', best_cv_score)
    print('Mean CV score (precision):', np.mean(cross_val_scores))
else:
    print("No successful trials.")
