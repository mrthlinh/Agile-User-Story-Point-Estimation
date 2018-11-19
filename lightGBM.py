#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:46:27 2018

@author: bking
"""
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

from scipy import sparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error

data_csv = pd.read_csv("data_csv/data")

x = sparse.load_npz("features/tf_idf_matrix.npz")
y = data_csv.point
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

lgbm = LGBMRegressor()
lgbm.fit(x_train,y_train)

y_pred = lgbm.predict(x_test)


print("Mean Absolute Error: ",mean_absolute_error(y_pred,y_test))
print("Median Absolute Error: ",median_absolute_error(y_pred,y_test))
print("Mean Squared Error: ",mean_squared_error(y_pred,y_test))