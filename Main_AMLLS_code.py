"""
Combined code for AMLLS project
"""
# Import
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score ,StratifiedGroupKFold , train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, make_scorer, balanced_accuracy_score
from sklearn.impute import KNNImputer
from sklearn.multiclass import OneVsRestClassifier
import random
import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import svm
from catboost import CatBoostClassifier

# import string
# import pickle
# import copy
# import glob
# import re
# from imblearn.over_sampling import SMOTENC
# Our classes and functions
import Functions_ML_Project
import Class_ML_Project

%load_ext autoreload
%autoreload 2
%aimport

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
IRRELEVANT_FEATURES = ["payer_code",'diag_1','diag_2','diag_3','repaglinide','nateglinide','chlorpropamide','tolbutamide','acarbose','miglitol','troglitazone',
 'tolazamide','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-pioglitazone',
 'admission_source_descriptor','admission_type_id','discharge_disposition_id','admission_source_id','patient_nbr']


# Define columns for OHE
OHE_regular_cols = ['race', 'gender', 'medical_specialty', 'insulin', 'diabetesMed', 'admission_type_descriptor',
                    'discharge_disposition_descriptor']
OHE_4_to_2_cols = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
diagnoses_cols = ['diag_1_cat', 'diag_2_cat', 'diag_3_cat']

# Note age feature will change to numerical later in the code!


# Define default models to initail test
models_defualt = {'Logisitic' : LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000),
          'XGBOOST' : XGBClassifier(use_label_encoder=False,random_state = 42, enable_categorical = True),
          'Tree' : DecisionTreeClassifier(random_state=42),
          'LGBM' : lgb.LGBMClassifier(random_state=42),
          'CatBoost' : CatBoostClassifier(random_seed = 42 , cat_features = ['age']),
                 'SVM': svm.SVC(kernel='linear',random_state=42, probability=True)}
random.seed(42)


# read the data file and gets train and test databases
db_train_df, db_test_df = Functions_ML_Project.clean_data_and_return_train_and_test('diabetic_data.csv')

# features pre-processing
id_names, mapping_dict = Functions_ML_Project.preform_ids_maping('IDS_mapping.csv')

# replace the disease code by disease name
training_df_new = Functions_ML_Project.apply_mapping(db_train_df, id_names, mapping_dict)

#
training_df_new = Functions_ML_Project.feature_engineering(training_df_new)

# Define the pipeline

pipeline = Pipeline([('feature_remover', Class_ML_Project.FeatureRemover(features_to_remove = IRRELEVANT_FEATURES)),
                     ('imputer_race', Class_ML_Project.DataFrameImputer(strategy='constant', fill_value='other', columns = ['race'])),
                     ('imputer_medical', Class_ML_Project.DataFrameImputer(strategy='most_frequent',columns = ['medical_specialty'])),
                     ('age_encoder', Class_ML_Project.MultiColumnLabelEncoder(columns=['age'])),
                     ('numerical_scaler',Class_ML_Project.NumericalTransformer(columns=NUMERICAL)),
                     ('OHE', Class_ML_Project.CustomOHEncoder(OHE_regular_cols= OHE_regular_cols, OHE_4_to_2_cols=OHE_4_to_2_cols,
                       change_col='change', diag_cols=diagnoses_cols))])

# Fit and transform the DataFrame
training_clean_imputed = pipeline.fit_transform(training_df_new)

# Get the removed column names
removed_column_names = pipeline.named_steps['feature_remover'].features_to_remove


# Finally print the results
print("DataFrame head after feature selection and imputation:")
print(training_clean_imputed.head())
print("DataFrame shape after feature selection and imputation:")
print(training_clean_imputed.shape)



# remove readmitted above 30 days 
training_clean_imputed = training_clean_imputed[training_clean_imputed.readmitted.isin(['<30', 'NO'])]

# run default models and compare with cross validation  
X = training_clean_imputed.drop('readmitted',axis  = 1 )

y = training_clean_imputed['readmitted']
y = LabelEncoder().fit_transform(y)
multi_model_cv = Class_ML_Project.MultiModelCV(models=models_defualt,
                                               score=balanced_accuracy_score,
                                               balance_threshold = 0.3)
multi_model_cv.fit(X, y)
defualt_models_original_dataframe = multi_model_cv.get_results()
print(defualt_models_original_dataframe)

#defualt_models_original_dataframe.to_csv('default_models_original_dataframe.csv', index=False)


# --------------------------------------------------------------- #
# run pipeline and defualt models on oversampled data by SMOTE-NC #
# --------------------------------------------------------------- #

post_smote_traning_dataset = pd.read_csv('post_smote_traning_dataset.csv',index_col= 0)

pipeline_smote = Pipeline([('imputer_race', Class_ML_Project.DataFrameImputer(strategy='constant', fill_value='other', columns = ['race'])),
                     ('imputer_medical', Class_ML_Project.DataFrameImputer(strategy='most_frequent',columns = ['medical_specialty'])),
                     ('age_encoder', Class_ML_Project.MultiColumnLabelEncoder(columns=['age'])),
                     ('OHE', Class_ML_Project.CustomOHEncoder(OHE_regular_cols= OHE_regular_cols, OHE_4_to_2_cols=OHE_4_to_2_cols,
                       change_col='change', diag_cols=diagnoses_cols))])

training_clean_smote_imputed = pipeline_smote.fit_transform(post_smote_traning_dataset)

# run default models and compare with cross validation  
X_smote = training_clean_smote_imputed.drop('readmitted',axis  = 1 )

y_smote = training_clean_smote_imputed['readmitted']
y_smote = LabelEncoder().fit_transform(y_smote)

multi_model_cv.fit(X_smote, y_smote)
defualt_models_smote_dataframe = multi_model_cv.get_results()
print(defualt_models_smote_dataframe)

#defualt_models_smote_dataframe.to_csv('default_models_smote_dataframe.csv', index=False)

# --------------------------------------------------------------- #
#   run pipeline and defualt models on oversampled data by GANs   #
# --------------------------------------------------------------- #

# --------------------------------- #
#                CTGAN              #
# --------------------------------- #


IRRELEVANT_FEATURES_for_GAN = ['repaglinide','nateglinide','chlorpropamide','tolbutamide','acarbose','miglitol','troglitazone',
 'tolazamide','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-pioglitazone',
 'admission_source_descriptor']

pipeline_CTGAN = Pipeline([('feature_remover', Class_ML_Project.FeatureRemover(features_to_remove = IRRELEVANT_FEATURES_for_GAN)),
                     ('imputer_race', Class_ML_Project.DataFrameImputer(strategy='constant', fill_value='Other', columns = ['race'])),
                     ('imputer_medical', Class_ML_Project.DataFrameImputer(strategy='most_frequent',columns = ['medical_specialty'])),
                     ('age_encoder', Class_ML_Project.MultiColumnLabelEncoder(columns=['age'])),
                     ('numerical_scaler',Class_ML_Project.NumericalTransformer(columns=NUMERICAL)),
                     ('OHE', Class_ML_Project.CustomOHEncoder(OHE_regular_cols= OHE_regular_cols, OHE_4_to_2_cols=OHE_4_to_2_cols,
                       change_col='change', diag_cols=diagnoses_cols))])

id_fold = pd.read_csv('id_fold.csv')

GAN_synthesized_data = [f for f in os.listdir() if 'CTGAN' in f and os.path.isfile(os.path.join(f))]

results_GAN_df = pd.DataFrame()
for fold,syn_data in enumerate(GAN_synthesized_data):
    GAN_train_fold, GAN_test_fold = Functions_ML_Project.GAN_data_preprocessing(syn_data,
                                                                                 training_clean_imputed, IRRELEVANT_FEATURES_for_GAN,
                                                                                 id_fold,OHE_regular_cols,fold+1,'max',pipeline_CTGAN)
    
    model_results = Functions_ML_Project.run_models_with_GAN(GAN_train_fold, GAN_test_fold, models = models_defualt)
    temp_df = pd.DataFrame([model_results])
    results_GAN_df = pd.concat([results_GAN_df, temp_df], ignore_index=True)
    
#results_GAN_df.to_csv('default_models_CTGAN_dataframe.csv', index=False)