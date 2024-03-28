import os
os.chdir('D:\yahel\phd\courses\ML in biology research\\final_project')


import numpy as np
import pandas as pd
import random
import string
import pickle
import copy
import glob
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import AllKNN

from sklearn.model_selection import StratifiedGroupKFold , train_test_split, cross_val_score , StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from Diabetic_readmission_essentials import categorize_dia, calculate_ratio
from Diabetic_readmission_essentials import ColumnRemover, NumericalTransformer ,SMOTENC_NS, CustomeLabelEncoder , CustomOHEncoder

from IPython.display import display
random.seed(42)

# read database #
path_to_data = os.path.join('data', 'diabetic_data.csv')
whole_data_df = pd.read_csv(path_to_data)
display(whole_data_df.head())

# Find for each feature the value it has
possibale_val_list = [whole_data_df[x].unique() for x in whole_data_df.columns]
possibale_val_df = pd.DataFrame(zip(list(whole_data_df.columns),possibale_val_list))
display(possibale_val_df)

# replace ?/None/Unknown/Invalid  with np.nan
whole_data_df[whole_data_df == "?"] = np.nan
whole_data_df[whole_data_df == "None"] = np.nan
whole_data_df[whole_data_df == "Unknown/Invalid"] = np.nan

# how many var and obs in data, data types and non -missing values
mis = whole_data_df.isna().sum()
whole_data_df.info()

# pop out the features with 1 option only
for fiture in whole_data_df[whole_data_df.columns[2:]]:
    if len(set(whole_data_df[fiture])) == 1:
        whole_data_df.pop(fiture)
    elif mis[fiture] > len(whole_data_df):
        whole_data_df.pop(fiture)

diagnosis_cols_list = ['diag_' + str(num) for num in [1,2,3]]

# ------------------------------------------ #
# Filter for patient diagnosed with diabetes #
# ------------------------------------------ #

# Regex pattern to locate diabeteic pateints 
pattern = r'250(\.\d{2})?'

# count diabetes diagnosis appearance  in all 3 daignosis columns 
diag_count = whole_data_df[diagnosis_cols_list].apply(lambda x: x.str.contains(pattern)).sum(axis =1)

diabetic_only_df = whole_data_df[diag_count > 0]
print('\n\n %d encounters out of %d total encounters are of diabetic pateints \n i.e. ~ %.2f%% '\
    %(diabetic_only_df.shape[0],whole_data_df.shape[0],
      (diabetic_only_df.shape[0]/whole_data_df.shape[0])*100))

# determine repeated encounter by the same patient #

duplicated_encounters = sum(diabetic_only_df['patient_nbr'].duplicated())
print('%d encounters are patient repeated ones, ~%.2f%%' % \
    (duplicated_encounters,
    (duplicated_encounters/diabetic_only_df.shape[0]*100)))


print('\n How many pateints with their repeated encounters \n',diabetic_only_df['patient_nbr'].value_counts().value_counts(),'\n')

# get the ratio of labels 
print('### Leabel ratios ###\n____________________________\n', calculate_ratio(diabetic_only_df['readmitted']), '\n____________________________\n')

""" Since a large portion of the encounters is a duplication of the same patient 
    We decided to treat each encounter as a different patient
    Therefore for test-train split we will follow the group_stratified split """

##############################
#   group_stratified split   #
##############################

# stratify by pateints' ID as group 
groups_for_stratification = diabetic_only_df['patient_nbr']
X = diabetic_only_df.drop('readmitted',axis=1)
Y = diabetic_only_df['readmitted']

# instentiate the StratifiedGroupKFold class
sgkf = StratifiedGroupKFold(n_splits=5) 

# use the CV split to create 5 spilt of a 80-20 ratio
cv_division = sgkf.split(X, Y,groups_for_stratification)

# unpack the first one - randomly 
train_test_index_array, *_ = cv_division

train_df = diabetic_only_df.iloc[train_test_index_array[0]]
test_df = diabetic_only_df.iloc[train_test_index_array[1]]

display(pd.DataFrame({'Complete dataset' : calculate_ratio(diabetic_only_df['readmitted']),
                      'Train datset (80%)' : calculate_ratio(train_df['readmitted']),
                      'Test datset (20%)' : calculate_ratio(test_df['readmitted'])}))

print('\n\nThere are %d pateint with overlap between train-test cohorts' % sum(np.in1d(train_df['patient_nbr'].unique(), test_df['patient_nbr'].unique())),'\n\n')

################################
#   features pre-processing    #
################################

# ------------------- #
# preform IDS-mapping #
# ------------------- #
mapping_path = 'D:\yahel\phd\courses\ML in biology research\\final_project\data\IDS_mapping.csv'

#Read readmission mapping file
mapping_df = pd.read_csv(mapping_path,header=None)
#print(mapping_df.head())
original_shape_num_rows = mapping_df.shape[0]

#Remove empty rows
mapping_df = mapping_df.dropna(how='all', inplace=False) #removes 2 gap rows

#Fill empty desctioption cells with NaNs they were previously NULL
mapping_df = mapping_df.fillna(value='NaN', inplace=False)
print("Does the new mapping_df has less rows than the original? ",mapping_df.shape[0]< original_shape_num_rows) #QC

#Reset indices
mapping_df = mapping_df.reset_index(drop=True)

### Rearrange IDS_mapping table by storing each categorical column with its description in a df inside a dictionary
#Find rows containing the string "id" in any column
id_rows = mapping_df[mapping_df.apply(lambda row: row.astype(str).str.contains('_id', case=False)).any(axis=1)]
id_names = list(id_rows[0])

# Extract the index of the id rows
id_rows_indices = list(id_rows.index)

# Create a dictionary with the 3 mapping dataframes
mapping_dictionary = {}

#Iterate over id_names to create subsets of mapping_df
for id_i in range(0,len(id_rows_indices)-1):
    start_index = id_rows_indices[id_i]
    end_index = id_rows_indices[id_i+1]
    
    # Extract the subset DataFrame
    current_id_df = mapping_df.iloc[start_index:end_index]  #+1 Include end_index
   
    #Assign the values of the first row as column names
    current_id_df.columns = current_id_df.iloc[0]

    # Drop the first row after using it as column names
    current_id_df = current_id_df[1:].reset_index(drop=True)

    # Assign the subset DataFrame to the corresponding key in the dictionary
    mapping_dictionary[id_names[id_i]] = current_id_df
    
# Add the last subset DataFrame explicitly
last_start_index = id_rows_indices[-1]
last_id_df = mapping_df.iloc[last_start_index:]
last_id_df.columns = last_id_df.iloc[0]
last_id_df = last_id_df[1:].reset_index(drop=True)
mapping_dictionary[id_names[-1]] = last_id_df

print("\n\nThe dictionary contains",len(mapping_dictionary.items()), "items\n\n")

### Replace the values in the training df with the values in the dictionary

training_df_new = train_df.copy()

training_df_new[id_names] = training_df_new[id_names].astype('str')

# Check that the columns in the DataFrame appear in the list
matching_columns = [col for col in training_df_new.columns if col in id_names]
matching_columns == id_names #QC

# Iterate over rows in training_df_new
for id_i in id_names:
#Extract the df in mapping_dictionary for the the id_column in training_df_new
    current_dict_df = mapping_dictionary[id_i]
    #Merge the discionary columns with the training_df_new
    training_df_new = training_df_new.merge(current_dict_df,on = id_i, how ='left',suffixes=['_left','_right'])

#Rename the new columns - last 3 columns in the df
new_column_names = [element[:-3] + "_descriptor" if element.endswith("_id") else element for element in id_names]

# Create a mapping dictionary for renaming
rename_dict = dict(zip(training_df_new.columns[-3:], new_column_names))
training_df_new = training_df_new.rename(columns=rename_dict)
print(training_df_new.iloc[:,-3:])

# ------------------------------------------------- #
# convert  diagnosis values from numbers to strings # 
# ------------------------------------------------- #

training_df_new['diag_1_cat'] = training_df_new['diag_1'].apply(categorize_dia)
training_df_new['diag_2_cat'] = training_df_new['diag_2'].apply(categorize_dia)
training_df_new['diag_3_cat'] = training_df_new['diag_3'].apply(categorize_dia)


# removes ages 0-10 as there are little of them and no readmision rate
training_df_new = training_df_new[~training_df_new['age'].isin(['[0-10)'])]

# removes values with Trauma Center and Newborn in the admission as they have only few (5) records
training_df_new = training_df_new[~training_df_new['admission_type_descriptor'].isin(['Trauma Center','Newborn'])]


##################################
#   oversampling using SMOTE-C   #
##################################

# column to remove after EDA conclusions
unecessery_columns = ['encounter_id', 'patient_nbr','weight','payer_code','diag_1','diag_2','diag_3',
                       'chlorpropamide','tolbutamide','tolbutamide','repaglinide','acarbose','miglitol','troglitazone','acetohexamide','nateglinide',
                       'tolazamide','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone',
                       'admission_type_id','discharge_disposition_id','admission_source_id','max_glu_serum','A1Cresult','admission_source_descriptor']

remover = ColumnRemover(columns_to_remove=unecessery_columns)
omitted_df = remover.transform(training_df_new)

nan_columns = training_df_new.groupby('readmitted').apply(lambda x: x.isna().sum())
print('\n# --------------------------------------------------------- #\n# \
    two features had above 50% NaNs values and were dropped #\
        \n# --------------------------------------------------------- #')
display(nan_columns[['race','medical_specialty','max_glu_serum','A1Cresult','diag_1_cat','diag_2_cat','diag_3_cat']])

# scale numerical values 
to_scale_cols = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications',
                 'number_outpatient','number_emergency','number_inpatient','number_diagnoses']

num_processing = NumericalTransformer(columns=to_scale_cols)

print('\n\n Scaling numerical features \n\n')
omitted_df = num_processing.fit_transform(omitted_df)


# labels encode 
print('\n\n Encode target labels \n\n')
X , y= CustomeLabelEncoder(label_column='readmitted').fit_transform(omitted_df)

# oversample with SMOT-NC
print('\n# --------------------------------------------------------- #\n# \
    Run oversampling with no size restrication using SMOT-NC #\
        \n# --------------------------------------------------------- #')
categorical_features_indices = [X.columns.get_loc(col) for col in X.select_dtypes(include=['category', 'object']).columns]

# add sampling_strategy = {0: 12000, 1: 12000} to restrict oversampling number
smote_os = SMOTENC_NS(categorical_features= categorical_features_indices,
                       k_neighbors= 5, seed=42)

X_resampled, y_resampled = smote_os.fit_resample(X,y)

# one-hot encode features
OHE_regular_cols = ['race','gender','age','medical_specialty','insulin','diabetesMed','admission_type_descriptor','discharge_disposition_descriptor']
OHE_4_to_2_cols = ['metformin','glimepiride', 'glipizide', 'glyburide', 'pioglitazone','rosiglitazone']
diagnoses_cols = ['diag_1_cat','diag_2_cat','diag_3_cat']

print('\n\n One Hot Encode features \n\n')
ohe = CustomOHEncoder(OHE_regular_cols= OHE_regular_cols, OHE_4_to_2_cols=OHE_4_to_2_cols,
                       change_col='change', diag_cols=diagnoses_cols)


X_ohe = ohe.fit_transform(X_resampled)
X_ohe.columns = X_ohe.columns.str.replace('[','')

# test numerous on defualt parameters with 10-fold CV to test best models
print('\n# ------------------------------------------------------------------------- #\n# \
   Test for numerous models on default parameters to chose best candidate models #\
        \n# ------------------------------------------------------------------------- #')
 
print('\n\n Run multinomial logistic regression \n CV score by average F1 measure\n\n')
# run logistic regression with CV of 10 folds
MLR = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=10000)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
mlr_scores = cross_val_score(MLR, X_ohe, y_resampled, cv=cv, scoring='f1_macro', verbose=3) 
print('CV score of 10 fold in logistic regression yeilds an avarge F1 averages score of %.2f' % mlr_scores.mean())

# run XGboost in sklearn version
xgb = XGBClassifier(use_label_encoder=False,)
xgb_scores = cross_val_score(xgb, X_ohe, y_resampled, cv=cv, scoring='f1_macro', verbose=3) 

print('CV score of 10 fold in XGBoost yeilds an avarge F1 averages score of %.2f' % xgb_scores.mean())
