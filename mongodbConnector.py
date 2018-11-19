#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:38:43 2018

@author: bking
"""


from pymongo import MongoClient
import pandas as pd
#import tensorflow as tf
#tf.enable_eager_execution()

# Connect to MondoDB and retrieve the data
class MongodbConnector():
    
    def __init__(self, host='localhost', port=27017,dbname="mydb"):
        self.host  = host
        self.port  = port
        self.client = MongoClient(host, port)
        self.db = self.client[dbname]
    
    def setCollection(self,collectionName):
        self.collection = self.db[collectionName]
        
    def getCollection(self):
        return self.collection
     
    def findProject(self,projectName):
        queryString = "^"+projectName
        return self.collection.find({'issuekey':{"$regex":queryString}})
    
    def getUnique(self,key):
        return self.collection.find().distinct(key)
    
    def getUniqueProject(self):
        projectName = self.getUnique("issuekey")
        projectName_ = [i.split("-")[0] for i in projectName]
        projectName_set = set(projectName_)
        return projectName_set
    
#    def getData(self,projectName):
#        project = self.findProject(projectName)
#        record = []
#        for document in project:
#            name = projectName
#            title = document["title"]
#            description = document['description']
#            storypoint = document['storypoint']
#            record.append([name,title,description,storypoint])
#        return record

    def getData(self):
        projectList = self.getUniqueProject()
        record = []
        for project in projectList:
            for document in self.findProject(project):
                name = project
                title = document["title"]
                description = document['description']
                storypoint = document['storypoint']
                record.append([name,title,description,storypoint])
        return record

        
# Test
#con = MongodbConnector()
#con.setCollection("storypoint")
#
#data = con.getData()



#dataset = tf.data.Dataset.zip((sentences, labels))


#words = tf.contrib.lookup.index_table_from_file("data/words.txt", num_oov_buckets=1)
#tags = tf.contrib.lookup.index_table_from_file("data/tags.txt")

def train_input_fn():
    pass

def eval_input_fn():
    pass


def tolowerCase():
    pass

#def preProcessing(self):
#    '''
#        Description: 
#            - Load data
#            - Convert from text to index
#            - 0 post-padding 
#        Usage:
#    '''
#    
##        sentence_size = 200
##    embedding_size = 50
#
#    
#    # we assign the first indices in the vocabulary to special tokens that we use
#    # for padding, as start token, and for indicating unknown words
#    
#    pad_id = 0
#
#         
#    print("Pad sequences (samples x time)")
#    self.x_train = sequence.pad_sequences(self.x_train_variable, 
#                                     maxlen=self.sentence_size,
#                                     truncating='post',
#                                     padding='post',
#                                     value=pad_id)
#    self.x_test = sequence.pad_sequences(self.x_test_variable, 
#                                    maxlen=self.sentence_size,
#                                    truncating='post',
#                                    padding='post', 
#                                    value=pad_id)
#    
#    print("x_train shape:", self.x_train.shape)
#    print("x_test shape:", self.x_test.shape)
#    


    
