#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:39:39 2018

@author: bking
"""
import pandas as pd
import pickle

model_list = ["CB","LGBM","RF"]

#feature_name = 'doc2vec'
feature_name = 'word2vec'

embedding_size = [10,50,100,300,500,1000,2000]

result_summary = pd.DataFrame()

for model in model_list:
    filename  = "results/"+model+"_"+feature_name
    with open(filename,'rb') as f:
        data = pickle.load(f)
    
    tmp = pd.DataFrame.from_dict(data)
    result_summary[model] = tmp['MSE']    

    
#for size in embedding_size:
#    Word2VecFeature(embedding_size=size)
#    result_each,result_total = RandomForestmodel()
#    
#    for k in record.keys():
#        record[k].append(result_total[k])

#result_summary = pd.DataFrame.from_dict(a)
result_summary.index = embedding_size
#result_summary = result_summary.drop(columns='size')
#result_summary.index.name = "Embedding Size"
ax = result_summary.plot(kind='line',title = "Effect of Embedding Size on MSE")
ax.set_xlabel("Embedding Size")
ax.set_ylabel("MSE")
#result_summary.plot(kind='bar')
#