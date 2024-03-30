#Combined code for decision tree and LGBM with default parameters

##########################    Decision Tree     ##########################

#Notes
# I added here the immputation and the OHE to be able to run the tree
#I also converted age into numeric

import pandas as pd
import numpy as np
import os
from scipy.stats import randint, uniform
import pickle
import graphviz
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import StratifiedKFold
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

#Covert Age to numeric feature
# Define a function to extract the lower and upper bounds of the age range and calculate the average
def extract_age_range_and_average(age_range):
    lower, upper = map(int, age_range.strip('[]()').split('-'))
    return (lower + upper) / 2

# Convert the age column to numeric
real_data['age'] = real_data['age'].apply(extract_age_range_and_average)

# Convert the age column to numeric
real_data['age'] = pd.to_numeric(real_data['age'])

print(real_data['age'])

# Define categorical feature list
# Subset all categorical (removing the non informative medication)
CATEGORICAL = ['race', 'gender','medical_specialty','max_glu_serum', \
'A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride', \
'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose', \
'miglitol','troglitazone','tolazamide','insulin','glyburide-metformin',\
'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed',\
'admission_type_descriptor', 'discharge_disposition_descriptor','admission_source_descriptor']

# Define numerical feature list
NUMERICAL = ['age','time_in_hospital', 'num_lab_procedures','num_procedures','num_medications',\
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
    ('imputer_medical', SimpleImputer(strategy='most_frequent'))])

# Fit and transform the DataFrame
training_clean_imputed = pipeline_preprosessing.fit_transform(real_data)

#Get the removed column names
removed_column_names = pipeline_preprosessing.named_steps['feature_remover'].features_to_remove

# Get the original column names before transformation
original_column_names = [col for col in real_data.columns if col not in removed_column_names]

# Convert the NumPy array to a DataFrame
training_clean_imputed = pd.DataFrame(training_clean_imputed, columns=original_column_names)
training_clean_imputed.to_pickle("training_clean_imputed.pkl")

print("DataFrame head after feature selection and imputation:")
print(training_clean_imputed.head())
# print("DataFrame shape after feature selection and imputation:")
# print(training_clean_imputed.shape)

### Decision Tree deafault parameters
#New categorical list 
# Update the CATEGORICAL list by removing the removed_column_names
CATEGORICAL = [column for column in CATEGORICAL if column not in removed_column_names]

# Save the new CATEGORICAL list as a pickle file
with open('CATEGORICAL.pkl', 'wb') as f:
    pickle.dump(CATEGORICAL, f)

print(CATEGORICAL)

# Define preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), NUMERICAL),
        ('cat', OneHotEncoder(drop='first'), CATEGORICAL)
    ])

# Fit and transform data (preprocessing)
transformed_data = preprocessor.fit_transform(training_clean_imputed)

# Convert the sparse matrix to a dense array
transformed_data_dense = transformed_data.toarray()  # transformed_data is the sparse matrix

# Create a DataFrame with appropriate column names
transformed_df = pd.DataFrame(transformed_data_dense, columns=preprocessor.get_feature_names_out())
print(transformed_df)

column_names = transformed_df.columns

# Remove prefixes "cat__" and "num__" from column names
cleaned_column_names = [name.split('__')[-1] for name in column_names]

# Assign the cleaned column names back to the DataFrame
transformed_df.columns = cleaned_column_names

# Print the DataFrame with cleaned column names
print(transformed_df)
transformed_df.to_pickle("transformed_df_27032024.pkl")
# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Encode the labels

# Define 'X' to contain the features and 'y' to contain the label
X = transformed_df
y = real_data['readmitted']
y_encoded = label_encoder.fit_transform(y)

print("Shape of X:", X.shape)
print("Shape of y:", y_encoded.shape)

# DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth = 7,random_state=42) #Optional pruning - ccp_alpha=0.01

# Fit the DecisionTreeClassifier
tree_clf.fit(X, y_encoded)

# Perform cross-validation
# Initialize StratifiedKFold with 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define our scoring metrics 
scoring_metrics = ['precision_weighted', 'recall_weighted', 'f1_weighted'] #We focus on precision

# Initialize lists to store mean scores for each metric
mean_scores = []

# Perform cross-validation and calculate mean scores for each metric
for scoring_metric in scoring_metrics:
    scores = cross_val_score(tree_clf, X, y_encoded, cv=skf, scoring=scoring_metric)
    mean_score = np.mean(scores)
    mean_scores.append(mean_score)
    print(f'Mean score for {scoring_metric}: {round(mean_score,4)}')

# Inverse transform encoded labels to obtain original class labels
original_class_labels = label_encoder.inverse_transform(y_encoded)

# Set the font size and maximum depth
font_size = 6  # Adjust the font size as needed
max_depth = 3  # Set the maximum depth of the tree to be displayed

# Plot the decision tree using Matplotlib
# plt.figure(figsize=(20,10))  # Adjust the figure size as needed
# plot_tree(tree_clf, filled=True, feature_names=X.columns, class_names=original_class_labels.astype(str),
#           fontsize=font_size, max_depth=max_depth)
#plt.show()

##########################    LGBM    ##########################
# Define LightGBM model
lgbm_classifier = lgb.LGBMClassifier(random_state=42)

# Random Search for LGBM
param_distribs = {
'n_estimators': randint(low=1, high=200),
'feature_fraction_bynode': uniform(0.05, 1.0)
}
rnd_search = RandomizedSearchCV(lgbm_classifier, param_distributions=param_distribs, n_iter=10,
cv=5,scoring= 'average_precision', random_state=42) #Can also use 'neg_mean_squared_error'

rnd_search.fit(X, y_encoded)

cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print("Best parameters found:", rnd_search.best_params_)
    
# Extract feture importance and visualize
feature_importances = rnd_search.best_estimator_.feature_importances_

# Feature names
features = X.columns

# Sort features based on importances
indices = np.argsort(feature_importances)[::-1]

# Number of top features to display
top_n = 10  # Adjust this number as needed

# Plot only the top features
plt.figure(figsize=(10, 6))
plt.title('Top {} Feature Importances'.format(top_n))
plt.barh(range(top_n), feature_importances[indices][:top_n], color='b', align='center')
plt.yticks(range(top_n), [features[i] for i in indices][:top_n])
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()  # Invert y-axis to display highest importance at the top
plt.tight_layout()

# Save the plot
plt.savefig('Top_{}_RF_importances.png'.format(top_n))
plt.show()