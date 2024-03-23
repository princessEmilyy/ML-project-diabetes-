### Decision Tree deafault parameters
#Notes
# I added here the immputation and the OHE to be able to run the tree
#Issues: Feature importnce + CV didn't work for me

import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Read the training set
folder_path = "C:/Users/Shirik/Dropbox (Weizmann Institute)/Shiri/Courses/AMLLS/ML-project-diabetes-/data/data_split/"
os.chdir(folder_path)
os.getcwd()
file_path = "train_df_after_rows_filtration.csv"
real_data = pd.read_csv(folder_path+file_path,index_col=0)

# Count the number of NaNs in each column
nan_counts_before_imputation = real_data.isna().sum()
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
pipeline_preprosessing = Pipeline([
    ('feature_remover', FeatureRemover()), 
    ('imputer_race', SimpleImputer(strategy='constant', fill_value='other')),
    ('imputer_medical', SimpleImputer(strategy='most_frequent'))
])

# Fit and transform the DataFrame
training_clean_imputed = pipeline_preprosessing.fit_transform(real_data)

#Get the removed column names
removed_column_names = pipeline_preprosessing.named_steps['feature_remover'].features_to_remove

# Get the original column names before transformation
original_column_names = [col for col in real_data.columns if col not in removed_column_names]

# Convert the NumPy array to a DataFrame
training_clean_imputed = pd.DataFrame(training_clean_imputed, columns=original_column_names)

print("DataFrame head after feature selection and imputation:")
print(training_clean_imputed.head())
print("DataFrame shape after feature selection and imputation:")
print(training_clean_imputed.shape)

### Model
# Separate features and target variable
X = training_clean_imputed.drop(columns=['readmitted'])
#X = training_clean_imputed.iloc[:, [3,4,5,10,13,14,15]] # test with part of the features
y = training_clean_imputed['readmitted']

#New categorical list 
# Ypdate the CATEGORICAL list by removing the removed_column_names
CATEGORICAL = [column for column in CATEGORICAL if column not in removed_column_names]
print(CATEGORICAL)

# Define preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL),
        ('cat', OneHotEncoder(drop='first'), CATEGORICAL)
    ])

# Create a DecisionTree pipeline
pipeline_decision_tree = Pipeline([
    ('preprocessor', preprocessor),
    ('tree_clf', DecisionTreeClassifier(random_state=42))
])

# Train model using pipeline
pipeline_decision_tree.fit(X, y)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Access the trained DecisionTreeClassifier from the pipeline
tree_clf = pipeline_decision_tree.named_steps['tree_clf']

# Check feature importance
feature_importance = tree_clf.feature_importances_

# Concatenate X and y along the columns axis to include the target column
X_with_target = pd.concat([X, y], axis=1) # Now, X_with_target contains both the features and the target variable
print(X_with_target.shape)

# # Create a DataFrame to display feature importance
importance_df = pd.DataFrame({'Feature': X_with_target.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print or visualize feature importance
print(importance_df)

# #Evaluate the model
# Accuracy
# accuracy = accuracy_score(y_true, y_pred)
# print("Accuracy:", accuracy)

# # Precision
# precision = precision_score(y_true, y_pred, average='weighted')  # Change average as needed
# print("Precision:", precision)

# # Recall
# recall = recall_score(y_true, y_pred, average='weighted')  # Change average as needed
# print("Recall:", recall)

# # F1-score
# f1 = f1_score(y_true, y_pred, average='weighted')  # Change average as needed
# print("F1-score:", f1)

# # ROC-AUC score
# roc_auc = roc_auc_score(y_true, y_pred)  # Assumes binary classification
# print("ROC-AUC Score:", roc_auc)

# # Confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)
# ### Perform cross-validation
# cv_scores = cross_val_score(pipeline_decision_tree, X, y, cv=5, scoring='recall')

# # Print the cross-validation scores
# print("Cross-Validation Scores:", cv_scores)

# # Calculate the mean cross-validation score
# mean_cv_score = cv_scores.mean()
# print("Mean Cross-Validation Score:", mean_cv_score)

#Visualizing The Decision Tree
# export_graphviz(
# pipeline_decision_tree['tree_clf'],
# out_file="C:/Users/Shirik/Dropbox (Weizmann Institute)/Shiri/Courses/AMLLS/data_tree.dot",
# feature_names=training_clean_imputed.columns,
# class_names=training_clean_imputed['readmitted'],
# rounded=True,
# filled=True
# )

#This command line converts the .dot file to a .png image file:
# $ dot -Tpng iris_tree.dot -o iris_tree.png

#Don't run this
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=42,
#             splitter='best')
#tree_clf.predict_proba([[]])
