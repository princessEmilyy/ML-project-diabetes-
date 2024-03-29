"""
functions for the project 
"""

import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score ,StratifiedGroupKFold , train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import KNNImputer
import random
import string
import pickle
import copy
import glob
import re
from imblearn.over_sampling import SMOTENC

from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


######################################### Functions  ########################################

def calculate_ratio(nome:pd.Series, deno = None):
    """_summary_
    Args:
        nome (pd.Series): _description_
        deno (_type_, optional): _description_. Defaults to None.
    """
    if deno is None:
        deno = nome
    elif deno is not pd.Series:
       return()
    
    return(nome.value_counts() / deno.value_counts().sum())



# a function thats categorises the diagnosis based on table 2
def categorize_dia(code:pd.Series):
    
    """
    Args:
    code is a pandas series contaning number
    these should be converted to objects/strings
    
    """
    
    if pd.isnull(code):
        return None
    if isinstance(code, str) and re.match('250(\.\d{2})?', code):
        return 'Diabetes'
    try:
        code = float(code)
    except ValueError:
        return 'Other'  # If it's Exx-Vxx or can't be converted to float, classify as 'Other'

    if 390 <= code <= 459 or code == 785:
        return 'Circulatory'
    if 460 <= code <= 519 or code == 786:
        return 'Respiratory'
    if 520 <= code <= 579 or code == 787:
        return 'Digestive'
    if 800 <= code <= 999:
        return 'Injury'
    if 710 <= code <= 739:
        return 'Musculoskeletal'
    if 580 <= code <= 629 or code == 788:
        return 'Genitourinary'
    if 140 <= code <= 239:
        return 'Neoplasms'
    if code in [780, 781, 784] or (790 <= code <= 799) or (240 <= code <= 279 and code != 250) or \
       (680 <= code <= 709) or code == 782 or (1 <= code <= 139) or (290 <= code <= 319) or \
       (280 <= code <= 289) or (320 <= code <= 359) or (630 <= code <= 679) or (360 <= code <= 389) or \
       (740 <= code <= 759):
        return 'Other'
    return 'Other'


# Define a custom scoring function
def custom_avg_precision_score(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average='macro')



def defaults_model_results(X:pd.DataFrame, y:pd.Series, folds:int, score): 
    """_summary_

    Args:
        X (pd.DataFrame): features to predict y
        y (pd.Series): prediction label
        folds (int): nu,ber of folds to test
        score (string/scorer object): is string than must be of one exisiting in sklearn. if not than must be a scorer that works with sklearn 
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    #logistic
    MLR = OneVsRestClassifier(LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000))
    mlr_scores = cross_val_score(MLR, X, y, cv=cv, scoring=score, verbose=3)
    
    # XGBOOST
    xgb = OneVsRestClassifier(XGBClassifier(use_label_encoder=False,))
    xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring=score, verbose=3)
    
    # random forest
    RFclass = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42,max_depth = 7))
    rf_score =  cross_val_score(RFclass, X, y, cv=cv, scoring=score, verbose=3)
    
    mean_score = [mlr_scores.mean(),xgb_scores.mean(),rf_score.mean()]
    models = ['Logistic regression','XGBoost','RandomForest']
    
    return_df = pd.DataFrame({'Model' : models, 'Score' : mean_score})
    return_df = return_df.sort_values(by='Score', ascending=False)
    return(return_df)


# Define a function to extract the lower and upper bounds of the age range and calculate the average
def extract_age_range_and_average(age_range):
    lower, upper = map(int, age_range.strip('[]()').split('-'))
    return (lower + upper) / 2

