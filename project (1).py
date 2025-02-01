#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[2]:


df=pd.read_csv("C:\\Users\\smartech\\OneDrive\\Desktop\\pcos_prediction_dataset.csv")


# In[3]:


df


# In[4]:


print (df.info())


# In[5]:


print(df.describe())


# In[6]:


print(df.isnull().sum())


# In[7]:


print(f"Number of duplicates: {df.duplicated().sum()}")


# In[8]:


print(df.columns)


# In[9]:


numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


# In[10]:


categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


# In[31]:


for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# In[33]:


for col in categorical_cols:
    df[col] = df[col].astype('category')


# In[35]:


df = df[(np.abs(zscore(df[numeric_cols])) < 3).all(axis=1)]


# In[37]:


scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# In[39]:


df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])


# In[ ]:




