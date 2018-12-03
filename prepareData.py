#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:50:21 2018

@author: bking
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import pandas as pd
#from sklearn.model_selection import train_test_split


#from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from helper import split_data
#import tempfile

class prepareData():
    def __init__(self):
        
        with open("helper/dictionary.pickle","rb") as f:
            dictionary = pickle.load(f)
        
        vocab_size = len(dictionary)
    
        # Load data -> change to mongoDb Connector later
        data = pd.read_csv("data_csv/data")
        y = data.point
        
        with open("features/tf_idf_vectorizer.pickle","rb") as f:
            vectorizer = pickle.load(f)
        
        corpus = data[['project','concat']]
                
        analyzer = vectorizer.build_analyzer() 
        
        corpus['concat'] = corpus.apply(lambda x: analyzer(x[1]),axis=1)
             
        tmp = corpus.apply(lambda x: len(x[1]),axis=1)
        
        sentence_size = max(tmp)
        
        corpus['concat'] = corpus.apply(lambda x: [dictionary.get(i) for i in x[1]],axis=1)
        
        
        x_train, x_test, y_train, y_test = split_data(corpus, y, ratio=0.2)
        
        pad_id = 0        
        
        x_train = sequence.pad_sequences(x_train['concat'].values,
                                 maxlen=sentence_size,
                                 truncating='post',
                                 padding='post',
                                 value=pad_id)
        
        x_test = sequence.pad_sequences(x_test['concat'].values,
                                 maxlen=sentence_size,
                                 truncating='post',
                                 padding='post',
                                 value=pad_id)
        
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        
        
        
        
        
        
        
        
        