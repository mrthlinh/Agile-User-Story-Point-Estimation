#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:36:07 2018

@author: bking
"""

from Doc2VecFeature import Doc2VecFeature
from RandomForest import RandomForestmodel
from lightGBM import lightGBMmodel
from catb import catBoostmodel
from multiprocessing import Pool
import pandas as pd
import pickle
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--proc', default='16', type=str, help='Number of proccessor') 
parser.add_argument('--model_name', default='RF', type=str, help='Model Name') 
args = parser.parse_args()
model_name = args.model_name
proc = int(args.proc)

print("Set the embedding size")
embedding_size = [10,50,100,300,500,1000,2000]
#embedding_size = [10,50]

print("Check embedding file existence")
file_list = set(['features/doc2vec_'+str(i)+'.csv' for i in embedding_size])
csv_list = set(glob.glob("features/*.csv"))
check = file_list.issubset(csv_list)
if check:
    print("Feature file exist")
else:
    print("Extracture Features")
    with Pool(proc) as p:
        feature = p.map(Doc2VecFeature, embedding_size)
        
#model_name = 'RF'
#model_name = 'LGBM'
#model_name = 'CB'
#embedding_size = [10,50]

#record = {'MSE':[],'MAE':[],'MdAE':[]}
#size = 10


def classification(data):
    model_name = data[0]
    size = data[1]
    x = data[2]
    
    if model_name == 'RF':
        result_each,result_total = RandomForestmodel(size,x)    
    if model_name == 'LGBM':
        result_each,result_total = lightGBMmodel(size,x)    
    if model_name == 'CB':
        result_each,result_total = catBoostmodel(size,x)  
        
    result_total['model'] = model_name
    result_total['size']  = size
    return result_total

#a = pd.read_csv("features/doc2vec_10.csv",index_col=0,low_memory=False)
#result = classification(['RF','10',input_list[0]])

input_list = [pd.read_csv("features/doc2vec_"+str(size)+".csv",index_col=0,low_memory=False) for size in embedding_size]

print("Inference")
with Pool(proc) as p:

#    test = p.map(classification, zip(['RF','LGBM','CB'],[embedding_size]*3))
    length = len(embedding_size)
    result = p.map(classification, zip([model_name]*length , embedding_size,input_list))

filename = "results/"+ model_name + "_doc2vec"
with open(filename,'wb') as f:
    pickle.dump(result,f)

print("Save file to: ",filename)
print(result)

#with open(filename,'rb') as f:
#    a = pickle.load(f)
    
#for size in embedding_size:
#    Word2VecFeature(embedding_size=size)
#    result_each,result_total = RandomForestmodel()
#    
#    for k in record.keys():
#        record[k].append(result_total[k])
#
#result_summary = pd.DataFrame.from_dict(record)
#result_summary.index = embedding_size
#   
#result_summary.plot(kind='bar')
