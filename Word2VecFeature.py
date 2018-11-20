#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:20:21 2018

@author: bking
"""

from gensim.test.utils import common_texts,get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
import pandas as pd


path = get_tmpfile("word2vec.model")
corpus = pd.read_hdf('helper/corpus_hdf', key='abc', mode='r')

model = Word2Vec(corpus, size=400, window=3, min_count=1, workers=4)
model.train(corpus,len(corpus),epochs = 10)

model.save("helper/word2vec.model")

#    model = gensim.models.Word2Vec(
#        documents,
#        size=150,
#        window=10,
#        min_count=2,
#        workers=10)
#    model.train(documents, total_examples=len(documents), epochs=10)

features = []

for i in range(len(corpus)):    
    sentence = corpus.iloc[i]
    ave_vector = 0
    for s in sentence:
#        print(s)
        ave_vector += model.wv[s]
    
    ave_vector = ave_vector / model.vector_size
    
    features.append(ave_vector)
columns = ['feature'+str(i) for i in range(model.vector_size)]
df_word2vec = pd.DataFrame.from_records(features,columns=columns)
df_word2vec.to_csv("features/word2vec_ave.csv")

