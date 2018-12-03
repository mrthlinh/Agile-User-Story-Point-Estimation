#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 23:48:35 2018

@author: bking
"""


def split_data(data,y,ratio=0.2):
    data_ = data.copy() 
    data_['train_test'] = 'train'
    
    def label_data(row):
        total = row.shape[0]
        test_size = int(total * ratio)
        new_data = ['train']*total
        
#        print('total size: ',total,' test size: ',test_size)
        
        new_data[-test_size:] = ['test']*test_size
        row['train_test'] = new_data
        return row
          
    data_ = data_.groupby(by='project').apply(label_data)
      
    x_train = data_.loc[data_.train_test == 'train',data_.columns[:-1]]
    x_test = data_.loc[data_.train_test == 'test',data_.columns[:-1]]
    y_train = y.loc[data_.train_test == 'train']
    y_test = y.loc[data_.train_test == 'test']
 
#        x_train = corpus.loc[corpus.train_test == 'train',['project','concat']]
#        x_test = corpus.loc[corpus.train_test == 'test',['project','concat']]
#        y_train = y.loc[corpus.train_test == 'train']
#        y_test = y.loc[corpus.train_test == 'test']
        
    return x_train,x_test,y_train,y_test
    
#
#data = pd.read_csv("data_csv/data")
#
#y = data.point
#x = data.loc[:,data.columns.delete(3)]
#
#x_train, x_test, y_train, y_test = split_data(x, y, ratio=0.2)
