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
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, make_scorer, log_loss
from sklearn.impute import KNNImputer
import random
import string
import pickle
import copy
import glob
import re

from imblearn.over_sampling import SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import optuna 
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import RandomSampler
import xgboost as xgb


import Class_ML_Project

######################################### Functions  ########################################

def read_and_clean_data_(file_name: str):
    """
    This function read the data base into Dataframe and clean it (e.g., remove '?', None)
    :param file_name: database file name to read
    :return: a clean data base
    """
    # read database
    whole_data_df = pd.read_csv(file_name, index_col= 'encounter_id')
    print(whole_data_df.head())

    # Find for each feature the value it has
    possible_val_list = [whole_data_df[x].unique() for x in whole_data_df.columns]
    possible_val_df = pd.DataFrame(zip(list(whole_data_df.columns), possible_val_list))
    print(possible_val_df)

    # replace ?/None/Unknown/Invalid  with np.nan
    whole_data_df[whole_data_df == "?"] = np.nan
    whole_data_df[whole_data_df == "None"] = np.nan
    whole_data_df[whole_data_df == "Unknown/Invalid"] = np.nan

    return whole_data_df




def clean_data_and_return_train_and_test(file_name: str):
    """

    :param file_name: database file name to read
    :return: clean database split into train adn test data bases
    """
    # get the clean data base
    whole_data_df = read_and_clean_data_(file_name)
    # how many var and obs in data, data types and non -missing values
    mis = whole_data_df.isna().sum()
    whole_data_df.info()

    # pop out the features with 1 option only
    for fiture in whole_data_df[whole_data_df.columns[2:]]:
        if len(set(whole_data_df[fiture])) == 1:
            whole_data_df.pop(fiture)
        elif mis[fiture] > len(whole_data_df):
            whole_data_df.pop(fiture)

    diagnosis_cols_list = ['diag_' + str(num) for num in [1, 2, 3]]

    # ------------------------------------------ #
    # Filter for patient diagnosed with diabetes #
    # ------------------------------------------ #

    # Regex pattern to locate diabetic patients
    pattern = r'250(\.\d{2})?'

    # count diabetes diagnosis appearance  in all 3 daignosis columns
    diag_count = whole_data_df[diagnosis_cols_list].apply(lambda x: x.str.contains(pattern)).sum(axis=1)

    diabetic_only_df = whole_data_df[diag_count > 0]
    print('\n\n %d encounters out of %d total encounters are of diabetic patients \n i.e. ~ %.2f%% '
          % (diabetic_only_df.shape[0], whole_data_df.shape[0], 100 * diabetic_only_df.shape[0]/whole_data_df.shape[0]))

    # determine repeated encounter by the same patient #
    duplicated_encounters = sum(diabetic_only_df['patient_nbr'].duplicated())
    print('%d encounters are patient repeated ones, ~%.2f%%' %
          (duplicated_encounters, (duplicated_encounters/diabetic_only_df.shape[0] * 100)))

    print('\n How many pateints with their repeated encounters \n',
          diabetic_only_df['patient_nbr'].value_counts().value_counts(), '\n')

    # get the ratio of labels
    print('### Label ratios ###\n____________________________\n',
          calculate_ratio(diabetic_only_df['readmitted']),
          '\n____________________________\n')

    """ Since a large portion of the encounters is a duplication of the same patient 
        We decided to treat each encounter as a different patient
        Therefore for test-train split we will follow the group_stratified split """

    ##############################
    #   group_stratified split   #
    ##############################

    # stratify by patients' ID as group
    groups_for_stratification = diabetic_only_df['patient_nbr']
    X = diabetic_only_df.drop('readmitted', axis=1)
    Y = diabetic_only_df['readmitted']

    # instantiate the StratifiedGroupKFold class
    sgkf = StratifiedGroupKFold(n_splits=5,)

    # use the CV split to create 5 spilt of a 80-20 ratio
    cv_division = sgkf.split(X, Y, groups_for_stratification)

    # unpack the first one - randomly
    train_test_index_array, *_ = cv_division

    train_df = diabetic_only_df.iloc[train_test_index_array[0]]
    test_df = diabetic_only_df.iloc[train_test_index_array[1]]

    print(pd.DataFrame({'Complete dataset': calculate_ratio(diabetic_only_df['readmitted']),
                        'Train dataset (80%)': calculate_ratio(train_df['readmitted']),
                        'Test dataset (20%)': calculate_ratio(test_df['readmitted'])}))

    print('\n\nThere are %d patient with overlap between train-test cohorts'
          % sum(np.in1d(train_df['patient_nbr'].unique(), test_df['patient_nbr'].unique())),
          '\n\n')

    return train_df, test_df




def preform_ids_maping(file_name: str):
    """
    preform IDS-mapping
    :param file_name:
    :return:
    """
    # Read readmission mapping file
    mapping_df = pd.read_csv(file_name, header=None)
    original_shape_num_rows = mapping_df.shape[0]

    # Remove empty rows
    mapping_df = mapping_df.dropna(how='all', inplace=False)  # removes 2 gap rows

    # Fill empty description cells with NaNs they were previously NULL
    mapping_df = mapping_df.fillna(value='NaN', inplace=False)
    print("Does the new mapping_df has less rows than the original? ",
          mapping_df.shape[0] < original_shape_num_rows)  # QC

    # Reset indices
    mapping_df = mapping_df.reset_index(drop=True)

    # Rearrange IDS_mapping table by storing each categorical column with its description in a df inside a dictionary
    # Find rows containing the string "id" in any column
    id_rows = mapping_df[mapping_df.apply(lambda row: row.astype(str).str.contains('_id', case=False)).any(axis=1)]
    name_id = list(id_rows[0])

    # Extract the index of the id rows
    id_rows_indices = list(id_rows.index)

    # Create a dictionary with the 3 mapping dataframes
    mapping_dictionary = {}

    # Iterate over id_names to create subsets of mapping_df
    for id_i in range(0, len(id_rows_indices) - 1):
        start_index = id_rows_indices[id_i]
        end_index = id_rows_indices[id_i+1]

        # Extract the subset DataFrame
        current_id_df = mapping_df.iloc[start_index:end_index]  # +1 Include end_index

        # Assign the values of the first row as column names
        current_id_df.columns = current_id_df.iloc[0]

        # Drop the first row after using it as column names
        current_id_df = current_id_df[1:].reset_index(drop=True)

        # Assign the subset DataFrame to the corresponding key in the dictionary
        mapping_dictionary[name_id[id_i]] = current_id_df

    # Add the last subset DataFrame explicitly
    last_start_index = id_rows_indices[-1]
    last_id_df = mapping_df.iloc[last_start_index:]
    last_id_df.columns = last_id_df.iloc[0]
    last_id_df = last_id_df[1:].reset_index(drop=True)
    mapping_dictionary[name_id[-1]] = last_id_df

    print("\n\nThe dictionary contains", len(mapping_dictionary.items()), "items\n\n")

    return name_id, mapping_dictionary




def apply_mapping(in_db: pd.DataFrame, name_id: list, mapping: dict):
    """
    replace the disease code by disease name
    :param in_db: input database to modify
    :param name_id: disease code
    :param mapping: dictionary that maps disease code by name
    :return: modify DF
    """
    # Replace the values in the training df with the values in the dictionary
    index_save = in_db.index
    df_new = in_db.copy()
    df_new[name_id] = df_new[name_id].astype('str')

    # Iterate over rows in df_new
    for ind_i in name_id:
        # Extract the df in mapping (dict) for the the id_column in df_new
        current_dict_df = mapping[ind_i]
        # Merge the dictionary columns with the df_new
        df_new = df_new.merge(current_dict_df, on=ind_i, how='left', suffixes=['_left', '_right'])

    # Rename the new columns - last 3 columns in the df
    new_column_names = [element[:-3] + "_descriptor" if element.endswith("_id") else element for element in name_id]

    # Create a mapping dictionary for renaming
    rename_dict = dict(zip(df_new.columns[-3:], new_column_names))
    df_new = df_new.rename(columns=rename_dict)
    print(df_new.iloc[:, -3:])
    df_new.index = index_save
    return df_new




def feature_engineering(in_df: pd.DataFrame):
    """
    Orginize the database so it will be more suitable for the ML
    :param in_df: input dataframe to work on
    :return: engineered database
    """
    # convert diagnosis values from numbers to strings #
    for columns in ['diag_1_cat', 'diag_2_cat', 'diag_3_cat']:
        in_df[columns] = in_df[columns.split('_cat')[0]].apply(categorize_dia)

    # removes ages 0-10 as there are little of them and no re-admission rate
    in_df = in_df[~in_df['age'].isin(['[0-10)'])]

    # removes values with Trauma Center and Newborn in the admission as they have only few (5) records
    in_df = in_df[~in_df['admission_type_descriptor'].isin(['Trauma Center', 'Newborn'])]

    """ 
        We hypotehsize that the model might benefit from converting age from categorical feature
        with ranges into a an ordinal feature. This is because there is a an ordinal and
        directional meaning to ages (as opposed to race or gender). Therefore we use LabelEncoder

    """

    return in_df




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
def custom_avg_precision_score(y_true, y_prob):
    # Assuming y_prob is a 2D array of shape [n_samples, n_classes]
    return average_precision_score(y_true, y_prob, average='macro')



# Define a function to extract the lower and upper bounds of the age range and calculate the average
def extract_age_range_and_average(age_range):
    lower, upper = map(int, age_range.strip('[]()').split('-'))
    return (lower + upper) / 2



# define a function that removes OHE sparse data
def remove_sparse_OHE(df_:pd.DataFrame, regex_colnames:list):
    sparse_OHE_cols = []
    for reg in regex_colnames:
        tmp_df = df_.filter(regex = reg)
        sparse_OHE_cols += tmp_df.columns[tmp_df.sum(axis = 0) / len(tmp_df) < 0.01].tolist()
    print('removed %i sparse OHE columns' % len(sparse_OHE_cols))
    df_.drop(sparse_OHE_cols, axis =1, inplace = True)
    return df_



# function to process GAN synthezied data and combine to real data 
def GAN_data_preprocessing(GAN_path:str, original_data:pd.DataFrame,filtration_col:list,
                          ID_df:pd.DataFrame,OHE_cols:list,num_fold:int,oversample,pre_pipeline):
    """
    GAN_path - a path to read the GAN synethisized data from
    original_data - training dataset which the GAN was run on
    filtration_col - the non relevant columns for the matrix to be removed
    OHE_cols - OHE columns to check for sparsity and than remove
    ID_df - dataframe to map the original ids for folds
    oversample - either "max" or a float to state the oversampling ratio (if max than raises the num of samples to be equal)
    pre_pipeline - preprceossing pipeline to run the new synethisized data through before combining with the fold data
    num_fold -which fold number to oversample datafrom
    """
    
    current_synthesized_df = pd.read_csv(GAN_path, index_col=0)
    
    current_synthesized_df_imputed = pre_pipeline.fit_transform(current_synthesized_df)
    current_synthesized_df_imputed = remove_sparse_OHE(current_synthesized_df_imputed,OHE_cols)

    # remove columns that were not existing in original data 
    missing_columns_from_original = current_synthesized_df_imputed.columns[~current_synthesized_df_imputed.columns.isin(original_data.columns)]
    current_synthesized_df_imputed.drop(missing_columns_from_original, axis = 1, inplace = True)

    # add columns that were not exisiting in GAN data only with 0 values
    missing_columns_GAN_df = pd.DataFrame(0,index = np.arange(current_synthesized_df_imputed.shape[0]), columns = original_data.columns[~original_data.columns.isin(current_synthesized_df_imputed.columns)])
    current_synthesized_df_imputed = pd.concat([current_synthesized_df_imputed.reset_index(drop = True),missing_columns_GAN_df],axis = 1)

    # read Id mapping for folds
    ID_df = ID_df[ID_df.fold_num == num_fold]

    # subset the training/testing fold
    subset_train = original_data[original_data.index.isin(ID_df.encounter_id[ID_df.Train_Val == 'Train'])]
    subset_test = original_data[original_data.index.isin(ID_df.encounter_id[ID_df.Train_Val == 'Val'])]

    # encod label for subsets
    subset_train.readmitted = LabelEncoder().fit_transform(subset_train.readmitted)
    subset_test.readmitted = LabelEncoder().fit_transform(subset_test.readmitted)

    # set sample size to oversample
    if oversample == 'max':
        n_OE = subset_train.readmitted.value_counts()[1] - subset_train.readmitted.value_counts()[0]
    elif isinstance(oversample, float):
        n_OE = round((subset_train.readmitted.value_counts()[1] - subset_train.readmitted.value_counts()[0]) * oversample)
    elif oversample == 0:
        n_OE = 0

    # oversample with GAN
    sampled_df = current_synthesized_df_imputed.sample(n=n_OE, random_state=42)
    subset_train = pd.concat([subset_train, sampled_df],axis = 0)
    
    return subset_train,subset_test



# run models on all folds of GAN
def run_models_with_GAN(train_df:pd.DataFrame,test_df:pd.DataFrame,
                         models:dict,balance_threshold:float =0.3):

    X_train = train_df.drop('readmitted' , axis = 1)
    y_train = train_df['readmitted']

    X_test = test_df.drop('readmitted' , axis = 1)
    y_test = test_df['readmitted']
    
    _, counts = np.unique(y_train, return_counts=True)
    ratios = counts / np.max(counts)
    is_balanced = sum(ratios > balance_threshold) == len(counts)

    # Dynamically select and add RandomForestClassifier based on balance
    rf_model_name = 'BalancedRandomForestClassifier' if not is_balanced else 'RandomForestClassifier'
    rf_model = BalancedRandomForestClassifier(random_state=42) if not is_balanced else RandomForestClassifier(random_state=42)
    models[rf_model_name] = rf_model

    results = {}
    for name, model in models.items():
        print("\n_____________\n",name,"\n_____________\n")
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        tmp_score = balanced_accuracy_score(y_test,y_pred)
        print('\n Score.................................. = %.3f\n' % tmp_score)
        results[name] = tmp_score
    return results




# funcitons that runs a preprocess SMOTE oversampling based on Folds give by a dataframe
def SMOTE_data_preprocessing(original_data:pd.DataFrame,filtration_col:list,
                          ID_df_:pd.DataFrame,OHE_regular_cols:list,
                             num_fold:int,pre_pipeline, post_pipeline):
    """
    original_data - training dataset which the GAN was run on
    filtration_col - the non relevant columns for the matrix to be removed

    ID_df - dataframe to map the original ids for folds
    pre_pipeline - preprceossing pipeline to run  data through before SMOTE-NC
    post_pipeline - preprceossing pipeline to run the new synethisized data through after oversampling
    OHE_regular_cols - list of OHE regular columns to check for sparsity
    num_fold -which fold number to oversample datafrom
    """
    
    original_data = pre_pipeline.fit_transform(original_data)

    
    # Oversample with SMOTE-NC #
    ID_df = ID_df_
    ID_df = ID_df[ID_df.fold_num == num_fold]

    # subset the training/testing fold
    subset_train = original_data[original_data.index.isin(ID_df.encounter_id[ID_df.Train_Val == 'Train'])]
    subset_test = original_data[original_data.index.isin(ID_df.encounter_id[ID_df.Train_Val == 'Val'])]

    # split for oversampling
    X_smote_train = subset_train.drop('readmitted', axis = 1)
    y_smote_train = subset_train['readmitted']

    # get categorical columns locs
    categorical_features_indices = [X_smote_train.columns.get_loc(col) for col in X_smote_train.select_dtypes(include=['category', 'object']).columns]
    smote_os = Class_ML_Project.SMOTENC_NS(categorical_features= categorical_features_indices,k_neighbors= 5, seed=42)

    # label encode for SMOTE
    label_encoder = LabelEncoder()
    y_encoded_smote_train = label_encoder.fit_transform(y_smote_train)
    subset_test.loc[:,'readmitted'] = label_encoder.fit_transform(subset_test.loc[:,'readmitted'])
    
    # change dype of target to int64
    subset_test['readmitted'] = subset_test['readmitted'].astype('int64')
    
    # oversample and combine
    X_resampled_smote, y_resampled_smote = smote_os.fit_resample(X_smote_train,y_encoded_smote_train)
    training_post_smote = pd.concat([X_resampled_smote, pd.Series(y_resampled_smote)],axis =1)
    training_post_smote.columns.values[-1] = 'readmitted'
    
    
    # run the post_pipeline to complete preprocssing before training
    training_post_smote = post_pipeline.fit_transform(training_post_smote)
    training_post_smote = remove_sparse_OHE(training_post_smote,OHE_regular_cols)
    
    subset_test = post_pipeline.fit_transform(subset_test)
    subset_test = remove_sparse_OHE(subset_test,OHE_regular_cols)
    
    # drop extra cols that are not in train test but are in test set
    test_drop_cols = subset_test.columns[~subset_test.columns.isin(training_post_smote.columns)]
    subset_test.drop(test_drop_cols, axis = 1,inplace = True)
    
    # add columns that were not exisiting in GAN data only with 0 values
    missing_columns_smote_df = pd.DataFrame(0,index = np.arange(subset_test.shape[0]), columns = training_post_smote.columns[~training_post_smote.columns.isin(subset_test.columns)])
    subset_test = pd.concat([subset_test.reset_index(drop = True),missing_columns_smote_df],axis = 1)
    
    # order columns the same as in training
    subset_test = subset_test[training_post_smote.columns]
    return training_post_smote,subset_test




# functions for hyperparameter tuning XGBoost
# only tune scale_pos to check of partial oversampling is better
def objective_tune_scale_pos_XGB(trial, dtrain, params_train, eval_metric, nfold, folds_idx): 
    
    params_search = {
                      "scale_pos_weight":trial.suggest_float("scale_pos_weight", 0.0, 1.0)
                    }
    
    params_train.update(params_search)
    
    #num_boost_round = params_train.pop('num_estimators')

    num_boost_round = 500
    
    cls_cv = xgb.cv(params=params_train, num_boost_round = num_boost_round, dtrain=dtrain,
                    nfold=nfold, shuffle=True,  metrics=eval_metric,
                    stratified=True,  seed=42, folds = folds_idx) 
    
    return (cls_cv.iloc[-1, 2])




# objective to tune other hyperparameters in steps - step 1 features are more important       

    { "num_estimators" :trial.suggest_int("num_estimators", 780, 820), 
                       "max_depth" : trial.suggest_int('max_depth', 3, 4),
                       "learning_rate" :trial.suggest_float("learning_rate", 0.15, 0.17), 
                       "subsample" :trial.suggest_float("subsample", 0.8, 0.85),
                       "gamma" :trial.suggest_float("gamma", 2, 3),
                       "min_child_weight":  trial.suggest_int('min_child_weight', 1, 2)
                    }

    
    params_train.update(params_search)
    
    num_boost_round = params_train.pop('num_estimators')
    
    cls_cv = xgb.cv(params=params_train, num_boost_round = num_boost_round, dtrain=dtrain,
                    nfold=nfold, shuffle=True, metrics=eval_metric,
                    stratified=True,  seed=42, folds = folds_idx) 
   
    return (cls_cv.iloc[-1, 2]) 




def objective_tune_stage_2(trial, dtrain, params_train, num_boost_round, eval_metric, nfold, folds_idx): 

    params_search = {   
                       "min_child_weight": trial.suggest_int("min_child_weight", 5, 15), 
                       "lambda": trial.suggest_float("lambda", 3.0, 6.0),
                       "alpha": trial.suggest_float("alpha", 2.5, 4.0),
                       "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 0.9),
                       "colsample_bynode" : trial.suggest_float("colsample_bynode", 0.95, 1.0),
                       "max_delta_step": trial.suggest_int("max_delta_step", 2, 7),
                       "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise"]),
                       "sampling_method": trial.suggest_categorical("sampling_method", ["uniform"]),
                       "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.65, 0.75),
                       "max_leaves": trial.suggest_int("max_leaves", 90, 110)
                    }
    
    
    params_train.update(params_search)
    
   
    cls_cv = xgb.cv(params=params_train, num_boost_round = num_boost_round, dtrain=dtrain,
                    nfold=nfold, shuffle=True,  metrics=eval_metric,  stratified=True,  seed=42, folds = folds_idx) 
    
    return (cls_cv.iloc[-1, 2])  


def print_best_callback(study, trial):
            print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")




# function that tunes hyperparameters      
def tune_hyperparameters_XGB(base_param,OE_data, fold_index,objective,
                             tune_steps = 'one' ,eval_metric = 'logloss',
                         device = 'gpu',  nfolds = 5, n_trials = 50, skip_steps= False):    
    device = device

    if device == 'gpu':
        tree_method = 'hist'
        cuda = True
    else:
        tree_method = 'auto'
        cuda = False 

    # set basic parameters 
    params = base_param

    # number of folds
    nfolds = nfolds

    # tuning trails for optuna
    n_trials = n_trials

    # set evaluation metric
    eval_metric = eval_metric

    if eval_metric == 'balanced_accuracy_score':
                 min_max_opt = 'maximize'
    elif  eval_metric == 'logloss' or eval_metric == 'error':
        min_max_opt = 'minimize'

    dtrain = xgb.DMatrix(OE_data.drop('readmitted', axis=1),
                         OE_data['readmitted'],enable_categorical=True)
    
        
    base_model = xgb.cv(params=base_param, dtrain=dtrain,
                    nfold=nfolds, shuffle=True,  metrics=eval_metric,
                    stratified=True,  seed=42, folds = fold_index)
    
    baseline_score = base_model[f"test-{eval_metric}-mean"].iloc[-1]
    
    balanced = OE_data.loc[fold_index[0][0]].readmitted.sum() == len(OE_data.loc[fold_index[0][0]])/2
    
    if balanced:
        return np.NAN, base_param ,baseline_score
    
    if tune_steps == 'one':
        study = optuna.create_study(direction=min_max_opt, sampler=RandomSampler(seed=0))
        func = lambda trial: objective(trial, dtrain, params, eval_metric, nfolds, fold_index)
        study.optimize(func, n_trials= n_trials,  show_progress_bar=True, callbacks=[print_best_callback]) 
        best_params = study.best_params
        params.update(best_params)
    else:
    # tune parameters in few stages 
        if not skip_steps:
            # 1st round of params 
            study = optuna.create_study(direction=min_max_opt, sampler=RandomSampler(seed=0))
            func = lambda trial: objective_tune_stage_1(trial, dtrain, params, eval_metric, nfolds, fold_index)
            study.optimize(func, n_trials= n_trials,  show_progress_bar=True, callbacks=[print_best_callback]) 
            best_params = study.best_params
            params.update(best_params)

            # 2nd round of params
            num_estimators = params.pop('num_estimators')
            study = optuna.create_study(direction=min_max_opt, sampler=RandomSampler(seed=0))
            func = lambda trial: objective_tune_stage_2(trial, dtrain, params, num_estimators, eval_metric, nfolds, fold_index)
            study.optimize(func, n_trials= n_trials,  show_progress_bar=True, callbacks=[print_best_callback]) 
            best_params = study.best_params
            params.update(best_params)
            params['num_estimators'] = num_estimators
        else:
            num_estimators = params.pop('num_estimators')
            study = optuna.create_study(direction=min_max_opt, sampler=RandomSampler(seed=0))
            func = lambda trial: objective_tune_stage_2(trial, dtrain, params, num_estimators, eval_metric, nfolds, fold_index)
            study.optimize(func, n_trials= n_trials,  show_progress_bar=True, callbacks=[print_best_callback]) 
            best_params = study.best_params
            params.update(best_params)
            params['num_estimators'] = num_estimators

    #print('num_estimators', num_boost_round)
    print('params', params)
    print('Best value', study.best_value)
    
    return study.best_value,params, baseline_score





# define objective for optuna optimization for Random forest
def objective_rf(trial, _df, folds,eval_metric,initial_params ={}):
    # Suggest parameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 35),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 1000),
    }
    
    # List to store the scores for each fold
    scores = []
    
    # Iterate over each fold
    for train_idx, test_idx in folds:
        
        train_df = _df.loc[train_idx]
        test_df = _df.loc[test_idx]
        
        X_train, X_test = train_df.drop('readmitted', axis = 1), test_df.drop('readmitted',axis = 1)
        y_train, y_test = train_df['readmitted'], test_df['readmitted']
        
        # Initialize the RandomForestClassifier with suggested parameters
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        # Fit the model
        clf.fit(X_train, y_train)
        
        
        # Calculate the accuracy (or any other metric)
        if eval_metric == "logloss":
            # Predict on the test set
            y_pred = clf.predict_proba(X_test)[:,1]
            score = log_loss(y_test, y_pred)
        elif eval_metric == "balanced_accuracy_score":
            y_pred = clf.predict(X_test)
            score = balanced_accuracy_score(y_test, y_pred)
        
        scores.append(score)

    # Calculate the average score across all folds
    average_score = np.mean(scores)
    return average_score




# hyperparamer tuning for RandomForest
def tune_hyperparameters_rf(_df, eval_metric,folds,n_trials=50):
    
    # tuning trails for optuna
    n_trials = n_trials

    # set evaluation metric
    eval_metric = eval_metric

    if eval_metric == 'balanced_accuracy_score':
                 min_max_opt = 'maximize'
    elif  eval_metric == 'logloss' or eval_metric == 'error':
        min_max_opt = 'minimize'
    
    # ---------- #
    # base model #
    # ---------- #
    
    base_params = {'max_depth':10,
                   "n_estimators":100,
                   "min_samples_split":5,
                   "min_samples_leaf":3,
                   "random_state":42,
                  "max_features": 'sqrt',
                  "criterion":'log_loss',
                  }
    
    # set basic parameters 
    params = base_params
    
    scores_base =[]
    for train_idx, test_idx in folds:
        
        train_df = _df.loc[train_idx]
        test_df = _df.loc[test_idx]
        
        X_train, X_test = train_df.drop('readmitted', axis = 1), test_df.drop('readmitted',axis = 1)
        y_train, y_test = train_df['readmitted'], test_df['readmitted']
        
        # Initialize the RandomForestClassifier with suggested parameters
        clf = RandomForestClassifier(**base_params,n_jobs = -1)
        
        # Fit the model
        clf.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = clf.predict(X_test)
        
        # Calculate the accuracy (or any other metric)
        if eval_metric == "logloss":
            # Predict on the test set
            y_pred = clf.predict_proba(X_test)[:,1]
            score = log_loss(y_test, y_pred)
        elif eval_metric == "balanced_accuracy_score":
            y_pred = clf.predict(X_test)
            score = balanced_accuracy_score(y_test, y_pred)
        
        scores_base.append(score)

    # Calculate the average score across all folds
    baseline_score = np.mean(scores_base)
    print(baseline_score)
    
    study = optuna.create_study(direction=min_max_opt, sampler=optuna.samplers.RandomSampler(seed=0))
    func = lambda trial: objective_rf(trial,_df,folds,eval_metric)
    study.optimize(func, n_trials=n_trials,
                   show_progress_bar=True,
                   callbacks=[print_best_callback])
    
    best_params = study.best_params
    base_params.update(best_params)

    print('params', params)
    print('Best value', study.best_value)
    
    return base_params, study.best_value,baseline_score


{'n_estimators': 611, 'max_depth': 10,
                    'min_samples_split': 10,
                    'min_samples_leaf': 2, 'max_leaf_nodes': 977}


#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "max_depth": trial.suggest_int("max_depth", 2, 35),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
#         "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 1000),
#         #"sampling_strategy": trial.suggest_float("sampling_strategy", 0.21, 1.0)
#     }

# objective for balanced random forest
def objective_brf(trial, X, y, n_splits,eval_metric, groupstrat):
    # Suggest parameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 600, 700),
        "max_depth": trial.suggest_int("max_depth", 8, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 8, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 950, 1200),
        #"sampling_strategy": trial.suggest_float("sampling_strategy", 0.21, 1.0)
    }
    
    clf = BalancedRandomForestClassifier(**params,replacement = True,
                                         bootstrap = True,
                                         random_state=42, n_jobs=-1,
                                        sampling_strategy = 'auto')
    
    # Stratified K-Fold cross-validator
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42,)
    
    # Calculate accuracy using cross-validation
    scores = cross_val_score(clf, X, y,groups = groupstrat ,cv=kf, scoring=eval_metric)
    
    # Return the negative mean accuracy
    return np.mean(scores)




# function for hyperparameter tuning for balanced random forest
def tune_hyperparameters_brf( X, y,n_splits, eval_metric,groupstrat,n_trials=50):
    
    # tuning trails for optuna
    n_trials = n_trials

    # set evaluation metric
    eval_metric = eval_metric
    
    # ---------- #
    # base model #
    # ---------- #
    
    base_param = {'replacement' : True,'bootstrap' : True,
                  'random_state' : 42,'sampling_strategy': 'auto'}
        
    # Initialize the RandomForestClassifier with suggested parameters
    clf = BalancedRandomForestClassifier(replacement = True,
                                       bootstrap = True,
                                       random_state=42,
                                         sampling_strategy = 'auto',
                                         n_jobs = -1)



    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42,)

    # Calculate accuracy using cross-validation
    base_scores = cross_val_score(clf, X, y,groups = groupstrat ,cv=kf, scoring=eval_metric,n_jobs=-1)

    # Calculate the average score across all folds
    mean_base_scores = np.mean(base_scores)
    print(mean_base_scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=0))
    func = lambda trial: objective_brf(trial,X,y,n_splits,eval_metric,groupstrat)
    study.optimize(func, n_trials=n_trials,
                   show_progress_bar=True,
                   callbacks=[print_best_callback])
    
    best_params = study.best_params

    print('params', base_param)
    print('Best value', study.best_value)
    if eval_metric == 'neg_log_loss':
         return params, -study.best_value,-mean_base_scores
    elif eval_metric == 'balanced_accuracy':
         return best_params, study.best_value,mean_base_scores
   