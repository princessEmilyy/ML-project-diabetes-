"""
for Eda of numerical, need to change to jupiter notebook.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np




NUMERIC_VAR = ['number_diagnoses','time_in_hospital','num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient',]

def boxplot_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range.
    outliers_bool = (df[col] <= Q1 - 1.5 * IQR) | (df[col] >= Q3 + 1.5 *IQR)
    return df[col][outliers_bool]


train_df = pd.read_csv("C:/Users/weitz/Documents/PhD/ML-project/train_cohort.csv")
print(train_df.head())

dict_outliers = {}


for col in NUMERIC_VAR:
    dict_outliers.update({col : boxplot_outliers(train_df,col)})


# Creating individual plots
cols = NUMERIC_VAR
for var in cols:
    sns.boxplot(x=var, data=train_df)
    plt.show()


for key, value in dict_outliers.items():
    print (key, len(value), np.unique(value))
