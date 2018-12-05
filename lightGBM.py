#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:46:27 2018

@author: bking
"""
from lightgbm import LGBMRegressor
import pandas as pd
#from sklearn.model_selection import train_test_split

from scipy import sparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error
from helper import split_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--size', default='100', type=str, help='Size of embedding') 
parser.add_argument('--feature_name', default='word2vec_ave', type=str, help='Feature Name word2vec_ave or doc2vec') 
args = parser.parse_args()
feature_name = args.feature_name

size = int(args.size)
x = pd.read_csv("features/"+feature_name+"_"+str(size)+".csv",index_col=0,low_memory=False)

print("============ Light Gradient Boosting Tree Regressor ======================")
print(parser.print_help())
print("==========================================================================")


def lightGBMmodel(size,x):
    data_csv = pd.read_csv("data_csv/data",low_memory=False)
    
    #x = sparse.load_npz("features/tf_idf_matrix.npz")
    
#    x = pd.read_csv("features/word2vec_ave_"+str(size)+".csv",index_col=0,low_memory=False)
    
    #x = pd.read_csv("features/doc2vec.csv",index_col=0)
    
    y = data_csv.point
    
    x_train, x_test, y_train, y_test = split_data(x, y, ratio=0.2)
    
    lgbm = LGBMRegressor(random_state=99)
    lgbm.fit(x_train.iloc[:,:-1],y_train)
    
    y_pred = lgbm.predict(x_test.iloc[:,:-1])
    
    
    result_total = dict()
    result_total['MSE'] = mean_squared_error(y_pred,y_test)
    result_total['MAE'] = mean_absolute_error(y_pred,y_test)
    result_total['MdAE']= median_absolute_error(y_pred,y_test)
    
    print("Mean Absolute Error: ",mean_absolute_error(y_pred,y_test))
    print("Median Absolute Error: ",median_absolute_error(y_pred,y_test))
    print("Mean Squared Error: ",mean_squared_error(y_pred,y_test))
    
    print("Evaluation on each project")
    tmp = pd.DataFrame()
    tmp['project'] = x_test['project']
    tmp['pred'] = y_pred
    tmp['truth'] = y_test
    
    result_each = tmp.groupby(by='project').apply(lambda col: mean_squared_error(col.pred,col.truth)).to_frame(name='MSE')
    result_each['MAE'] = tmp.groupby(by='project').apply(lambda col: mean_absolute_error(col.pred,col.truth))
    result_each['MdAE'] = tmp.groupby(by='project').apply(lambda col: median_absolute_error(col.pred,col.truth))
    
    return (result_each,result_total)
    
def main():
    lightGBMmodel(size,x)
    
if __name__ == "__main__":
    main()