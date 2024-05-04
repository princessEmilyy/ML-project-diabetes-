%load_ext autoreload
%autoreload 2
%aimport

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder

from sklearn.model_selection import cross_val_score ,StratifiedGroupKFold , train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.impute import KNNImputer

import random
from catboost import CatBoostClassifier

import string
import pickle
import copy
import glob
import re
from imblearn.over_sampling import SMOTENC
# Our classes and functions
import Functions_ML_Project
import Class_ML_Project

# Global variables
# Define categorical feature list
# Subset all categorical (removing the non informative medication)
CATEGORICAL = ['age','race', 'gender', 'medical_specialty', 'max_glu_serum',
               'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
               'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
               'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
               'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
               'diabetesMed', 'admission_type_descriptor', 'discharge_disposition_descriptor',
               'admission_source_descriptor']

# Define numerical feature list
NUMERICAL = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
             'number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient']

# Define irrelevant feature list
IRRELEVANT_FEATURES = ["payer_code",'diag_1','diag_2','diag_3','repaglinide',	
 'nateglinide','chlorpropamide','tolbutamide','acarbose','miglitol','troglitazone',
 'tolazamide','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-pioglitazone',
 'admission_source_descriptor','admission_type_id','discharge_disposition_id','admission_source_id','patient_nbr']


# read the data file and gets train and test databases
db_train_df, db_test_df = Functions_ML_Project.clean_data_and_return_train_and_test('diabetic_data.csv')

# features pre-processing
id_names, mapping_dict = Functions_ML_Project.preform_ids_maping('IDS_mapping.csv')

# replace the disease code by disease name
training_df_new = Functions_ML_Project.apply_mapping(db_train_df, id_names, mapping_dict)

#
training_df_new = Functions_ML_Project.feature_engineering(training_df_new)

pipeline = Pipeline([('feature_remover', Class_ML_Project.FeatureRemover(features_to_remove = IRRELEVANT_FEATURES)),
                     ('numerical_scaler',Class_ML_Project.NumericalTransformer(columns=NUMERICAL))])

training_pre_smote = pipeline.fit_transform(training_df_new)

# split for oversampling
X = training_pre_smote.drop('readmitted', axis = 1)
y = training_pre_smote['readmitted']

# label encode for SMOTE
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# get categorical columns
categorical_features_indices = [X.columns.get_loc(col) for col in X.select_dtypes(include=['category', 'object']).columns]

smote_os = Class_ML_Project.SMOTENC_NS(categorical_features= categorical_features_indices,k_neighbors= 5, seed=42)

X_resampled, y_resampled = smote_os.fit_resample(X,y_encdoded)

training_post_smote = pd.concat([X_resampled, pd.Series(label_encoder.inverse_transform(y_resampled))],axis=1)
training_post_smote.columns.values[-1] = 'readmitted'
training_post_smote.to_csv('post_smote_traning_dataset.csv')