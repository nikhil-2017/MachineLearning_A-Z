# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 02:36:25 2020

@author: Nikhil
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# Matrix of features
# Matrix/Vector that store independent and dependent values/data.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# Avoidiing dummy data/variable
X = X[:, 1:]


# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

X_opt = np.array(X[:, [0,1,2,3,4,5]], dtype = 'float')
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:, [0,1,3,4,5]], dtype = 'float')
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:, [0,3,4,5]], dtype = 'float')
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:, [0,3,5]], dtype = 'float')
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:, [0,3]], dtype = 'float')
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())