import catboost as cb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
import optuna
import pickle
from sklearn.model_selection import KFold


File = open('Training_data-tuple_x_y.pkl', 'rb')
X, y = pickle.load(File)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    break


def objective(trial):
    cb_params = {
        'iterations': 400,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.1, 1.0),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 100),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.1, 20.0),
        'random_strength': trial.suggest_float('random_strength', 1.0, 2.0),
        'depth': trial.suggest_int('depth', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
        "use_best_model": True,
        "task_type": "GPU",
        'random_seed': 42
    }

    model = cb.CatBoostClassifier(**cb_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, timeout=100)
print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)