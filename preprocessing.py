# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk import tokenize
import re
import pickle

from mongodbConnector import MongodbConnector
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk.stem.wordnet import WordNetLemmatizer
import nltk

from collections import Counter
from sklearn.model_selection import train_test_split

from scipy import sparse


#nltk.download('wordnet')

def expand_contractions(sentence, contraction_mapping): 
     
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),  
                                      flags=re.IGNORECASE|re.DOTALL) 
    def expand_match(contraction): 
        match = contraction.group(0) 
        first_char = match[0] 
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                        
        expanded_contraction = first_char+expanded_contraction[1:] 
        return expanded_contraction 
         
    expanded_sentence = contractions_pattern.sub(expand_match, sentence) 
    return expanded_sentence 

if __name__ == "__main__":
    
    # Get data from MongoDB
    DBmanager = MongodbConnector()
    DBmanager.setCollection("storypoint")
    data = DBmanager.getData()
    
    # Put it into Pandas
    pd_data = pd.DataFrame.from_records(data,columns=['project','title','user_story','point'])
    
    # list of project
    project_list = list(pd_data.project.unique())
    
    # Contraction Map
    with open('helper/contraction_map.pickle', 'rb') as f:
        CONTRACTION_MAP = pickle.load(f)
        
    # Lematization model
    with open('helper/WordNetLemmatizer.pickle', 'rb') as f:
        lmtzr = pickle.load(f)
    
    # Concatinate title and user_story
    pd_data['concat'] = pd_data.apply(lambda x: x[1] + x[2],axis=1)
    
    corpus = pd_data['concat']
    # Save CSV file for later use
    pd_data.to_csv("data_csv/data",index=False)
    
#    vectorizer = CountVectorizer(stop_words='english',strip_accents="ascii")
    vectorizer = TfidfVectorizer(stop_words='english',strip_accents="ascii")
    vectorizer.fit(corpus)
     
    # TF_IDF features
    with open('features/tf_idf_vectorizer.pickle', 'wb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        pickle.dump(vectorizer, f, pickle.HIGHEST_PROTOCOL)    
        
    tf_idf_features = vectorizer.transform(corpus)
    
    # TF_IDF features
    sparse.save_npz("features/tf_idf_matrix.npz", tf_idf_features)

    
#    with open('features/tf_idf.pickle', 'wb') as f:
#        # The protocol version used is detected automatically, so we do not
#        # have to specify it.
#        pickle.dump(tf_idf_features, f, pickle.HIGHEST_PROTOCOL)    
        
#    tokenizer = vectorizer.build_tokenizer()
#    preprocessing = vectorizer.build_preprocessor()
    analyzer = vectorizer.build_analyzer()    
    dictionary = vectorizer.vocabulary_
    
    # Save dictionary
    with open('helper/dictionary.pickle', 'wb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    
    # Tokenizer
#    token = corpus.apply(lambda x: tokenizer(x))
#    length_token = sum(token.apply(lambda x: len(x)))
    
    corpus = corpus.apply(lambda x: analyzer(x))
    
#    corpus = corpus.apply(lambda x: tokenizer(x))
    
    # Lemmatization -> Not using because IOS and IO are totally different
#    corpus = corpus.apply(lambda x: [lmtzr.lemmatize(i) for i in x])
    
    # Expand Contraction
    corpus = corpus.apply(lambda x: [expand_contractions(i,CONTRACTION_MAP) for i in x])
    
    corpus.to_hdf('helper/corpus_hdf',key='abc')
    
    
    
    
