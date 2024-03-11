####### This code performs feature selection and categorical feature imputation using a pipline #######
#Libraries
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
import os

# Folder path
folder_path = "C:/Users/Shirik/Dropbox (Weizmann Institute)/Shiri/Courses/AMLLS/ML-project-diabetes-/data/data_split/"
os.chdir(folder_path)
os.getcwd()

file_path = "train_df_after_rows_filtration.csv"
training_clean = pd.read_csv(folder_path+file_path,index_col=0)
print("The original training dataset shape: ",training_clean.shape)

# Count the number of NaNs in each column
nan_counts_before_imputation = training_clean.isna().sum()
#print(nan_counts_before_imputation)

# Define categorical feature list
# Subset all categorical (removing the non informative medication)
CATEGORICAL = ['race', 'gender', 'age','medical_specialty','max_glu_serum', \
'A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride', \
'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose', \
'miglitol','troglitazone','tolazamide','insulin','glyburide-metformin',\
'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed',\
'admission_type_descriptor', 'discharge_disposition_descriptor','admission_source_descriptor']

# Define numerical feature list
NUMERICAL = ['time_in_hospital', 'num_lab_procedures','num_procedures','num_medications',\
             'number_diagnoses','number_outpatient', 'number_emergency','number_inpatient']
# Note: there are no missing values in the numerical features so imputation will be done on the categorical only

##### Feature selection and Imputation using a pipline #####
"""There are reasons to exclude features: 
1. Irrelevant based on domain
2. >50%  missing values
3. Low variance in the feature cannot contribute to the model
"""
# Define a global variable with irrelevant features as list
IRRELEVANT_FEATURES = ["payer_code"]

# Define a class to remove features using fit and transform
class FeatureRemover(BaseEstimator, TransformerMixin): #Remove features with irellevant information 
    def __init__(self):
        self.features_to_remove = []
    
    def fit(self, X, y=None):
        # Remove features with >50%  missing values
        #if self.features_to_remove is None:
            # Calculate the percentage of missing values in each column
        missing_percentages = (X.isnull().sum() / len(X)) * 100
            # Filter out features (columns) where missing percentage is above 50%
        self.features_to_remove += missing_percentages[missing_percentages > 50].index.tolist()
        # Low variance
        # Check if each column has more than two unique values
        for column in X.columns:
            if len(X[column].unique()) < 2:
                self.features_to_remove.append(column)

        return self
    
    def transform(self, X):
        return X.drop(columns=self.features_to_remove)

# Define the pipeline
pipeline = Pipeline([
    ('feature_remover', FeatureRemover()), 
    ('imputer_race', SimpleImputer(strategy='constant', fill_value='other')),
    ('imputer_medical', SimpleImputer(strategy='most_frequent'))
])

# Fit and transform the DataFrame
training_clean_imputed = pipeline.fit_transform(training_clean)

#Get the removed column names
removed_column_names = pipeline.named_steps['feature_remover'].features_to_remove

# Get the original column names before transformation
original_column_names = [col for col in training_clean.columns if col not in removed_column_names]

# Convert the NumPy array to a DataFrame
training_clean_imputed = pd.DataFrame(training_clean_imputed, columns=original_column_names)

print("DataFrame head after feature selection and imputation:")
print(training_clean_imputed.head())
print("DataFrame shape after feature selection and imputation:")
print(training_clean_imputed.shape)

# ### Try a more sophicticated imputation - KNN
# # class CategoricalKNNImputer(TransformerMixin):
# #     def __init__(self, n_neighbors=5):
# #         self.n_neighbors = n_neighbors
# #         self.imputer = KNNImputer(n_neighbors=self.n_neighbors, weights='uniform')

# #     def fit(self, X, y=None):
# #         self.X = X.copy()
# #         self.cat_cols = X.select_dtypes(include=['object', 'category']).columns

# #         # One-hot encode categorical columns
# #         self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# #         self.encoder.fit(X[self.cat_cols])

# #         # Fit KNN imputer
# #         self.imputer.fit(self.encoder.transform(X[self.cat_cols]))

# #         return self

# #     def transform(self, X):
# #         X_transformed = X.copy()

# #         # One-hot encode categorical columns
# #         cat_encoded = pd.DataFrame(self.encoder.transform(X_transformed[self.cat_cols]), columns=self.encoder.get_feature_names_out(self.cat_cols), index=X_transformed.index)
# #         X_transformed = pd.concat([X_transformed.drop(columns=self.cat_cols), cat_encoded], axis=1)

# #         # Impute missing values
# #         X_transformed_imputed = pd.DataFrame(self.imputer.transform(X_transformed), columns=X_transformed.columns, index=X_transformed.index)

# #         # Restore original categorical columns
# #         X_restored = X_transformed_imputed.copy()
# #         for col in self.cat_cols:
# #             if col in X_restored.columns:
# #                 X_restored[col] = X[col]

# #         return X_restored

# # # Example KNN imputer:
# # # Initialize the imputer
# # imputer_knn = CategoricalKNNImputer(n_neighbors=5)

# # # Fit and transform your data
# # imputed_knn_fit_transform = imputer_knn.fit_transform(example_cat_df)
# # imputed_knn_df = pd.DataFrame(imputed_knn_fit_transform)
# # print(imputed_knn_df.columns)

# # print(imputed_knn_df.columns)

# # #Check if imputation worked:
# # print("Most freq impute:",df_imputed.isna().sum())
# # print("KNNimputed: ", imputed_knn_df.isna().sum())
# # print("Not Imputed: ", example_cat_df.isna().sum())