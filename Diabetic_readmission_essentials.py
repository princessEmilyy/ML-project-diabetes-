# function to calculate ratios

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

from sklearn.model_selection import StratifiedGroupKFold , train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer


from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import AllKNN




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

# Defines a class for EDA plots for clerity 
class plotsEDA:
    def __init__(self):
        pass  # This class does not require initialization of attributes for now

    def plot_pie_chart(self, data, title='Pie Chart', colors=None, ax=None,
                       title_font_size=24, annotation_font_size=18, pie_font_size=18,
                       annotation_offsets=None):
        """
        Plot a pie chart with customizable title, colors, and font sizes.

        Parameters:
        - data: pandas Series with index as categories and values as counts
        - title: string, title of the plot
        - colors: list of color names or serial numbers accepted by matplotlib
        - title_font_size: int, font size of the title
        - annotation_font_size: int, font size of the annotations outside the pie
        - pie_font_size: int, font size of the pie chart's text (percentages)
        - ax: matplotlib axes, which one to draw for multuple plots if wanted
        annotation_offsets: list of tuples, offsets for annotations in the form (x_offset, y_offset)
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
        else:
            fig = ax.figure
            

        def func(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            return "{:.1f}%".format(pct, absolute)

        wedges, texts, autotexts = ax.pie(data.values, autopct=lambda pct: func(pct, data.values),
                                          colors=colors if colors else ['#fdb147', 'lightcoral', 'lightblue'],
                                          textprops=dict(color="black", fontsize=pie_font_size))

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        # set off set so no overlap between labels
        if annotation_offsets is None:
            annotation_offsets = [(0.3, 0.3)] * len(wedges)  # Default offset for each annotation
        
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            offset = annotation_offsets[i] if i < len(annotation_offsets) else (0.3, 0.3) # if off set tupes are shorter than slices resort to deaults 
            xytext = (offset[0]*np.sign(x)*100, offset[1]*y*100)  # Adjusting multiplier for positioning
            
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            
            
            ax.annotate(data.index[i], xy=(x, y), xytext=xytext, textcoords='offset points',
                        horizontalalignment=horizontalalignment, fontsize=annotation_font_size, **kw)

        ax.set_title(title, fontsize=title_font_size)

        # only show plot when singular
        if ax is None:
            plt.show()
    
    pass
    
    def plot_boxplot(self, data, x, y, colors=['#fdb147', 'lightcoral', 'lightblue'],
                     showmeans=True, meanprops=None, ax=None):
        """
        Plot a seaborn boxplot with customizable options.

        Parameters:
        - data: DataFrame, the data source for plotting.
        - x: str, name of the column in `data` to be used as x-ticks (categorical).
        - y: str, name of the column in `data` for y-axis values.
        - colors: list, color palette for the boxplot.
        - showmeans: bool, whether to show the mean marker.
        - meanprops: dict, properties for the mean marker.
        - ax: matplotlib axes object, optional axes to plot on.
        """
        
        if meanprops is None:
            meanprops = {"markerfacecolor":"yellow", "markeredgecolor":"black"}

        if ax is None:
            sns.boxplot(data=data, x=x, y=y, palette=colors, showmeans=showmeans, meanprops=meanprops)
        else:
            sns.boxplot(data=data, x=x, y=y, palette=colors, showmeans=showmeans, meanprops=meanprops, ax=ax)
        
        if ax is None:  # Only show the plot if no external axes were provided
            plt.show()


# A class for dropping columns - for pipeline use
class ColumnRemover(BaseEstimator, TransformerMixin):
    
    """
    here fit does nothing as only transform is required to omit the wanted columns
    supply coulmn names that you wish to omit in the form of a list
    """
    
    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove
        
    def fit(self, X, y=None):
        # just return self
        return self
    
    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Remove the specified columns and return the modified DataFrame
        return X.drop(columns=self.columns_to_remove, errors='ignore')
    

# a class for numerical scaling
# optional log2 for chosen features by name
class NumericalTransformer(BaseEstimator, TransformerMixin):
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

# a class for label encoding
class CustomeLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,label_column:str = None):
        """class to encode the label and split the dataframe to y and X
            Args:
            label_column (str): the column name of the target column
            X (Dataframe): the dataframe to be split and encoded
        """

        self.label_column = label_column
        self.label_encoder = None
         
        if self.label_column is None:
            raise ValueError("Label column name must be specified.")
        
    def fit(self,X, y=None):
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(X[self.label_column])
        return self
    
    def transform(self, X):
        
        X_transformed = X.drop(columns=[self.label_column])
        y_transformed = pd.Series(self.label_encoder.transform(X[self.label_column]), name=self.label_column)
        
        return X_transformed, y_transformed
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    
# SMOTENC transformer for use in an sklearn pipeline.
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
        #returned_df = X_resampled
        #returned_df['label'] = label_encoder.inverse_transform(y_resampled) # get original labels value back
        #pd.DataFrame(returned_df)
        return X_resampled, y_resampled 
    
# costumize OHE class for pipeline
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

    def transform(self, X):
        result = X.copy()
        
        # Apply regular OHE
        if self.OHE_regular_cols:
            transformed = self.ohe.transform(result[self.OHE_regular_cols]).toarray()
            result = result.drop(columns=self.OHE_regular_cols)
            result = result.join(pd.DataFrame(transformed, columns=self.ohe.get_feature_names_out(), index=result.index))
        
        # Apply 4-to-2 encoding - for medication
        if self.OHE_4_to_2_cols:
            result[self.OHE_4_to_2_cols] = result[self.OHE_4_to_2_cols].replace({'No': 0, 'Steady': 0, 'Up': 1, 'Down': 1})
        
        # Apply "change" transformation 
        #based on medication swap and not dosage change
        
        if self.change_col and self.OHE_4_to_2_cols:
            # checks if dosage was not chamges among medication exisitng in dataset
            dosage_changed_bool = result[self.OHE_4_to_2_cols].apply(lambda x: sum(x > 0) == 0, axis=1)
            
            # iterate to over all records ,mark 1 where doasge was not changed but medication was
            new_change = [(1 if i and ch == 'Ch' else 0) for i, ch in zip(dosage_changed_bool, result[self.change_col])]
            result[self.change_col] = new_change
        
        # Apply disease diagnosis encoding
        if self.diag_cols and self.unique_diagnoses is not None:
            # prepare a zero matrix to count for dieases per patient
            ohe_diagnosis = pd.DataFrame(np.zeros((result.shape[0], len(self.unique_diagnoses))),
                                         columns=self.unique_diagnoses, index=result.index)
            #get disease diagnosis per record 
            record_disease = result[self.diag_cols].apply(lambda x: x.value_counts().index.values, axis=1)
            
            # iterate over pateint diagnosis and add one to ohe_diagnosis in the relevant place
            for row, diag in enumerate(record_disease):
                for dis in diag:
                    if dis in ohe_diagnosis.columns:
                        ohe_diagnosis.loc[row, dis] = 1
            
            # drop Diabetes since they all have it
            ohe_diagnosis.drop(['Diabetes'], axis=1, inplace=True, errors='ignore')
            ohe_diagnosis = ohe_diagnosis.iloc[:, :-1]  # Drop last column for None diagnosis
            
            # remove input columns
            result = result.drop(columns=self.diag_cols)
            # add the untouched columns
            result = result.join(ohe_diagnosis)
        
        return result
    
    

# AllKNN downsampler transformer for use in an sklearn pipeline
class AllKNNResampler(BaseEstimator, TransformerMixin):
    def __init__(self, label_column='label', n_neighbors=6, sampling_strategy='all', kind_sel='mode'):
        self.label_column = label_column
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.kind_sel = kind_sel
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        # For compatibility with pipeline, but actual fitting happens in fit_resample
        return self

    def transform(self, X, y=None):
        # For compatibility with pipeline Not used, as fit_resample handles the resampling.
        return X

    def fit_resample(self, X, y=None):
        
        y = X[self.label_column]
        X = X.drop(columns=self.label_column)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Initialize and apply AllKNN
        allknn = AllKNN(n_neighbors=self.n_neighbors, sampling_strategy=self.sampling_strategy, kind_sel=self.kind_sel)
        X_resampled, y_resampled = allknn.fit_resample(X, y_encoded)
        
        # Decode labels back to original form
        y_resampled = self.label_encoder.inverse_transform(y_resampled)
        
        # Combine the resampled features and labels into a DataFrame
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_df[self.label_column] = y_resampled
        
        return resampled_df