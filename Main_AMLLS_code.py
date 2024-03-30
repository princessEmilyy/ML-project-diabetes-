"""
Combined code for AMLLS project
"""
# Import
import pandas as pd
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
# import string
# import pickle
# import copy
# import glob
# import re
# from imblearn.over_sampling import SMOTENC
# Our classes and functions
import Functions_ML_Project
import Class_ML_Project

# Global variables
# Define categorical feature list
# Subset all categorical (removing the non informative medication)
CATEGORICAL = ['race', 'gender', 'age', 'medical_specialty', 'max_glu_serum',
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
IRRELEVANT_FEATURES = ["payer_code"]


# for OHE
OHE_regular_cols = ['race', 'gender', 'age', 'medical_specialty', 'insulin', 'diabetesMed', 'admission_type_descriptor',
                    'discharge_disposition_descriptor']
OHE_4_to_2_cols = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
diagnoses_cols = ['diag_1_cat', 'diag_2_cat', 'diag_3_cat']

# Note age feature will change to numerical later in the code!

# Models with defualt parameters to test prior to tuning - Randomforest is embbeded in the MultiModelCV class
models_defualt = {'Logisitic' : LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000),
          'XGBOOST' : XGBClassifier(use_label_encoder=False,random_state = 42),
          'Tree' : DecisionTreeClassifier(random_state=42),
          'LGBM' : lgb.LGBMClassifier(random_state=42)}

random.seed(42)


def read_and_clean_data_(file_name: str):
    """
    This function read the data base into Dataframe and clean it (e.g., remove '?', None)
    :param file_name: database file name to read
    :return: a clean data base
    """
    # read database
    whole_data_df = pd.read_csv(file_name)
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
          Functions_ML_Project.calculate_ratio(diabetic_only_df['readmitted']),
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
    sgkf = StratifiedGroupKFold(n_splits=5)

    # use the CV split to create 5 spilt of a 80-20 ratio
    cv_division = sgkf.split(X, Y, groups_for_stratification)

    # unpack the first one - randomly
    train_test_index_array, *_ = cv_division

    train_df = diabetic_only_df.iloc[train_test_index_array[0]]
    test_df = diabetic_only_df.iloc[train_test_index_array[1]]

    print(pd.DataFrame({'Complete dataset': Functions_ML_Project.calculate_ratio(diabetic_only_df['readmitted']),
                        'Train dataset (80%)': Functions_ML_Project.calculate_ratio(train_df['readmitted']),
                        'Test dataset (20%)': Functions_ML_Project.calculate_ratio(test_df['readmitted'])}))

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
    return df_new


def feature_engineering(in_df: pd.DataFrame):
    """
    Orginize the database so it will be more suitable for the ML
    :param in_df: input dataframe to work on
    :return: engineered database
    """
    # convert diagnosis values from numbers to strings #
    for columns in ['diag_1_cat', 'diag_2_cat', 'diag_3_cat']:
        in_df[columns] = in_df[columns.split('_cat')[0]].apply(Functions_ML_Project.categorize_dia)

    # removes ages 0-10 as there are little of them and no re-admission rate
    in_df = in_df[~in_df['age'].isin(['[0-10)'])]

    # removes values with Trauma Center and Newborn in the admission as they have only few (5) records
    in_df = in_df[~in_df['admission_type_descriptor'].isin(['Trauma Center', 'Newborn'])]

    """ 
        We hypotehsize that the model might benefit from converting age from categorical feature 
        with ranges into a numerical featue. The way we do it is by averaging the value of the lower and upper
        age. For example: [60-70) becomes 65 years old. This is because there is a an ordinal and 
        directional meaning to ages (as opposed to race or gender). Another assumption we make is that the 
        averaged age is a good apporximation biologically, the difference between 60 to 70 is probably 
        not so dramatic so by avaraging we still represent the reallity.
    """

    # Convert the age column to numeric
    in_df['age'] = in_df['age'].apply(Functions_ML_Project.extract_age_range_and_average)

    # Convert the age column to numeric
    in_df['age'] = pd.to_numeric(in_df['age'])

    return in_df


# read the data file and gets train and test databases
db_train_df, db_test_df = clean_data_and_return_train_and_test('diabetic_data.csv')

# features pre-processing
id_names, mapping_dict = preform_ids_maping('IDS_mapping.csv')

# replace the disease code by disease name
training_df_new = apply_mapping(db_train_df, id_names, mapping_dict)

#
training_df_new = feature_engineering(training_df_new)

# Define the pipeline
pipeline = Pipeline([('feature_remover', Class_ML_Project.FeatureRemover()),
                     ('imputer_race', SimpleImputer(strategy='constant', fill_value='other')),
                     ('imputer_medical', SimpleImputer(strategy='most_frequent'))
                     ])

# Fit and transform the DataFrame
training_clean_imputed = pipeline.fit_transform(training_df_new)

# Get the removed column names
removed_column_names = pipeline.named_steps['feature_remover'].features_to_remove

# Get the original column names before transformation
original_column_names = [col for col in training_df_new.columns if col not in removed_column_names]

# Convert the NumPy array to a DataFrame
training_clean_imputed = pd.DataFrame(training_clean_imputed, columns=original_column_names)

# Finally print the results
print("DataFrame head after feature selection and imputation:")
print(training_clean_imputed.head())
print("DataFrame shape after feature selection and imputation:")
print(training_clean_imputed.shape)

# multi_model_cv = MultiModelCV(models=models_defualt, score='average_precision',balance_threshold = 0.2)
# multi_model_cv.fit(X_ohe, y_resampled)
# display(multi_model_cv.get_results())