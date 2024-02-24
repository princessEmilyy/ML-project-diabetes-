# function to calculate ratios

import numpy as np
import pandas as pd
import random
import string
import pickle
import copy
import glob
from functools import partial
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold , train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

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