"""
class for the project
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


######################################### class ########################################

class FeatureRemover(BaseEstimator, TransformerMixin): 
    #Remove features with irellevant information 
    ##### Feature selection and Imputation using a pipline #####
    """There are reasons to exclude features: 
    1. Irrelevant based on domain
    2. >50%  missing values
    3. Low variance in the feature cannot contribute to the model
    """
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



class NumericalTransformer(BaseEstimator, TransformerMixin):
    # a class for numerical scaling
    # optional log2 for chosen features by name
    def __init__(self, columns=None, log_column='num_medications'):
        
        """_summary_
            columns: list of column names for Min-Max scaling
            log_column: specific column name to apply log2 transformation before Min-Max scaling
        """
        self.columns = columns
        self.log_column = log_column
        self.scalers = {}  # To store individual scalers per column
        
    def fit(self, X, y=None):
        # Fit scaler to each specified column individually
        for col in self.columns:
            scaler = MinMaxScaler()
            if col == self.log_column:
                # Log-transform then fit scaler
                self.scalers[col] = scaler.fit(np.log2(X[[col]] + 1))
            else:
                # Fit scaler directly
                self.scalers[col] = scaler.fit(X[[col]])
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            if col == self.log_column:
                # Log-transform then scale
                X_transformed[col] = self.scalers[col].transform(np.log2(X_transformed[[col]] + 1))
            else:
                # Scale directly
                X_transformed[col] = self.scalers[col].transform(X_transformed[[col]])
        return X_transformed



class SMOTENC_NS(BaseEstimator, TransformerMixin):
    
    def __init__(self, categorical_features,
                 sampling_strategy = 'auto', k_neighbors=5, seed =42):
        """
        Parameters:
        categorical_features: array of ints corresponding to the indices specifying the categorical features
        - sampling_strategy: Determine oversampling ratios, for multiclass dict is preferable.
            defualts to "auto" and will oversample all classes to the majority class
        - k_neighbors: Number of nearest neighbors used for the algorithm.
        - seed: Random pseudoseed for KNN
        """
        self.categorical_features = categorical_features
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.sampling_strategy = sampling_strategy
    
    def fit_resample(self, X, y):
        """
        Fits the SMOTENC resampler to the data and resamples it.
        
        Parameters:
        - X: Features matrix
        - y: Target vector
        
        Returns:
        - X_resampled: The resampled features matrix
        - y_resampled: The resampled target vector
        """
        
        
        if self.categorical_features is None:
            raise ValueError("Categorical feature indexes are not specified.")
        
        # Encode your target variable if it's categorical
        
        # Initialize SMOTENC with user-specified parameters
        self.smotenc = SMOTENC(categorical_features=self.categorical_features, 
                               k_neighbors=self.k_neighbors, 
                               random_state=self.seed,
                               sampling_strategy = self.sampling_strategy)
        
        # Fit SMOTENC and resample the data
        X_resampled, y_resampled = self.smotenc.fit_resample(X, y)

        return X_resampled, y_resampled 
    


class CustomOHEncoder(BaseEstimator, TransformerMixin):
    
    """_summary_ 
    OHE categorical features in different manners
    OHE_regular_cols - columns to reguular OHE with sklearn class
    OHE_4_to_2_cols - columns to change 4 values to 2 values
                      all medication were reduced to 1 - changed dose / 0 - stable/NaN
    change_col - column to chnage after chage in OHE_4_to_2_cols
                 specifically for "Change" column to see if there was a change in medication based
                 based on a "Yes" and a lack of change of dosage in other medications
    diag_cols - coulmns to be expanded specifically diagnoses columns where each pateint
                had more than 2 diagnoses so expand to column per disease and drops diabetes
    """
    def __init__(self, OHE_regular_cols=[], OHE_4_to_2_cols=[], change_col=None, diag_cols=[]):
        self.OHE_regular_cols = OHE_regular_cols
        self.OHE_4_to_2_cols = OHE_4_to_2_cols
        self.change_col = change_col
        self.diag_cols = diag_cols
        self.ohe = OneHotEncoder(drop='if_binary')
        self.unique_diagnoses = None

    def fit(self, X, y=None):
        # Fit the regular OHE encoder
        if self.OHE_regular_cols:
            self.ohe.fit(X[self.OHE_regular_cols])
        
        # Prepare unique diagnoses for diagnosis encoding
        if self.diag_cols:
            melted_disease = pd.melt(X[self.diag_cols])
            self.unique_diagnoses = melted_disease['value'].unique()
        
        return self


