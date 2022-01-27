# EDA_Light

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



#change categorical data of diagnosis column into numerical (0,1) 
df = pd.read_csv('cancer.csv')

label_encode = LabelEncoder()
labels = label_encode.fit_transform(df['diagnosis'])
df['target'] = labels
df.drop(columns = ['id','diagnosis'],axis=1,inplace=True)
df['target']



#check missing values (NO missing values for any column)
df.isnull().sum()
df.target.value_counts()

df.groupby(['target']).mean() # mean of any column is greater than median



#Exploratory Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='target',data=df)
# Create a distribution plot for every column (almost all distributions are right skewed)
for col in df:
    sns.displot(x=col,data=df)
    
    
    
    #Check the outliers (all columns have outliers in distribution)

for col in df:
    plt.figure()
    df.boxplot([col])
    
    
    
    #Correlation (some variables are highly correlated, then should be removed)
corr_matrix = df.corr()

plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,cbar=True,annot=True,cmap='Blues',fmt='.1f')








