#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:20:21 2018

@author: bking
"""

from gensim.test.utils import common_texts,get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd


path = get_tmpfile("my_doc2vec_model")
corpus = pd.read_hdf('helper/corpus_hdf', key='abc', mode='r')

def list2str(row):
    out_str = ""
    for i in row:
        out_str += " "+ i 
    return out_str
    
corpus = corpus.apply(lambda x: list2str(x))

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus.values)]

#model = Doc2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

model = Doc2Vec(documents, vector_size=200, min_count=1, workers=4)

#model.train(corpus,len(corpus),epochs = 10)

model.save("helper/doc2vec.model")

#    model = gensim.models.Word2Vec(
#        documents,
#        size=150,
#        window=10,
#        min_count=2,
#        workers=10)
#    model.train(documents, total_examples=len(documents), epochs=10)

features = [model.docvecs[i] for i in range(len(documents))]

columns = ['feature'+str(i) for i in range(model.vector_size)]
df_doc2vec = pd.DataFrame.from_records(features,columns=columns)
df_doc2vec.to_csv("features/doc2vec.csv")

#model.wv.most_similar(['ios'],topn=10)
#[('ios8', 0.7491174936294556),
# ('blackberry', 0.6112785339355469),
# ('ios9', 0.6080747842788696),
# ('ipad', 0.607194185256958),
# ('watch', 0.6058827638626099),
# ('ios7', 0.5971351861953735),
# ('simulators', 0.5963417291641235),
# ('watchkit2', 0.5910321474075317),
# ('watchos', 0.5894814729690552),
# ('testingduring', 0.5705143213272095)]