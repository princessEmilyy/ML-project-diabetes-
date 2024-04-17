"""
The aim of this file is to train AMLLS model with different parameters
The program gets the model and parameters from command line e.g.,
1. Python Initiate_model_AMLLS.py -f "AMLLS model outputs.txt" -m "RandomForest" -p "a:#22,c:d"
2. Python Initiate_model_AMLLS.py -f "AMLLS_model_outputs.csv" -m "Logisitic" -p "multi_class:multinomial,solver:lbfgs,max_iter:#10000"
"""
from argparse import ArgumentParser
import copy
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn import svm
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import Class_ML_Project as cl


# Define default models to initial test
# models_defualt = {'Logisitic': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000),
#                   'Tree': DecisionTreeClassifier(random_state=42),
#                   'LGBM': lgb.LGBMClassifier(random_state=42),
#                   'CatBoost': CatBoostClassifier(n_estimators=3, random_state=42),
#                   'SVM': svm.SVC(kernel='linear', random_state=42),
#                   'XGBOOST': XGBClassifier(use_label_encoder=False, random_state=42, enable_categorical=True)}
models_build_param_dict = {'Logisitic': LogisticRegression, 'Tree': DecisionTreeClassifier, 'LGBM': lgb.LGBMClassifier,
                           'CatBoost': CatBoostClassifier, 'SVM': svm.SVC, 'XGBOOST': XGBClassifier}


def build_dict_from_str(in_str:str):
    ret_dict = dict()
    for str_element in in_str.split(','):
        Key, Value = str_element.split(':')
        if Value.strip(' ')[0] == '#':
            Value = int(Value.strip(' ')[1:])
        ret_dict[Key.lstrip(' ')] = Value
    return ret_dict

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="file_name", help="write results to output FILE", metavar="FILE", default='')
parser.add_argument("-m", "--model", dest="model", help="Select the model name to run", metavar="MODEL",
                    default='RandomForest')  # shall runs BalancedRandomForestClassifier or RandomForestClassifier
parser.add_argument("-p", "--params", dest="params", help="Select the model parameters to use", metavar="PARAMs",
                    default=dict())
args = parser.parse_args()

output_file_name = args.file_name
model_name = args.model
params = build_dict_from_str(args.params)

File = open('Training_data-tuple_x_y.pkl', 'rb')
X, y = pickle.load(File)

if model_name == 'RandomForest':
    balance_threshold = 0.02
    _, counts = np.unique(y, return_counts=True)
    ratios = counts / np.max(counts)
    is_balanced = sum(ratios > balance_threshold) == len(counts)

    # Dynamically select and add RandomForestClassifier based on balance
    model_name = 'BalancedRandomForestClassifier' if not is_balanced else 'RandomForestClassifier'
    if not is_balanced:
        model_name = BalancedRandomForestClassifier(random_state=42)
    else:
        model_name = RandomForestClassifier(random_state=42)

_ = {model_name: models_build_param_dict[model_name](**params)}
multi_model_cv = cl.MultiModelCV(models=_, score='average_precision', balance_threshold=0.2)
multi_model_cv.fit(X, y)
results = multi_model_cv.get_results()
results.to_csv(output_file_name)

print(multi_model_cv.get_results())