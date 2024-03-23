'''We think that the model might benefit from converting age from categorical feature 
with ranges into a numerical featue. The way we do it is by averaging value of the lower and upper
age. For example: [60-70) becomes 65 years old. This is because there is a an ordinal and 
directional meaning to ages (as opposed to race or gender). Another assumption we make is that the 
averaged age is a good apporximation biologically, the difference between 60 to 70 is probably 
not so dramatic so by avaraging we still represent the reallity.
''' 
### Converting  age to numerical by taking the average value
import pandas as pd
import os

# Folder path
folder_path = "C:/Users/Shirik/Dropbox (Weizmann Institute)/Shiri/Courses/AMLLS/ML-project-diabetes-/data/data_split/"
os.chdir(folder_path)
os.getcwd()
file_path = "train_df_after_rows_filtration.csv"
real_data = pd.read_csv(folder_path+file_path,index_col=0)

# Define a function to extract the lower and upper bounds of the age range and calculate the average
def extract_age_range_and_average(age_range):
    lower, upper = map(int, age_range.strip('[]()').split('-'))
    return (lower + upper) / 2

# Convert the age column to numeric
real_data['age'] = real_data['age'].apply(extract_age_range_and_average)

# Convert the age column to numeric
real_data['age'] = pd.to_numeric(real_data['age'])

print(real_data['age'])