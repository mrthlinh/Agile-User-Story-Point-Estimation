#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:22:30 2018

@author: bking
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.enable_eager_execution()

#from movieReviewData import movieReviewData
from prepareData import prepareData
import argparse


tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

#################### Define some parameter here ###############################

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default= './models/ckpt/', type=str, help='Dir to save a model and checkpoints') 
parser.add_argument('--saved_dir', default='./models/pb/', type=str, help='Dir to save a model for TF serving')
parser.add_argument('--step_size', default=10, type=int, help='Step size')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
args = parser.parse_args()


print("================== Custom model in LSTM ===========================")
print(parser.print_help())
print("===================================================================")


############################### Load and Preprocesing data ###########################################
# Load data
data = prepareData()

# Preprocessing data
#data.preProcessing()

vocab_size = data.vocab_size
embedding_size = 100
sentence_size = data.sentence_size

# Prepare data
x_train = data.x_train
x_test = data.x_test

y_train = data.y_train
y_test  = data.y_test


def LSTM_model_fn(features, labels, mode):
    """
        Description: Custom model LSTM
        Usage:
        return: 
    """
    
    # [batch_size x sentence_size x embedding_size]    
    
    # Using embeddings, you can represent each category as a vector of floats of the desired size
    # which can be used as features for the rest of the model
    inputs = tf.contrib.layers.embed_sequence(
            features['x'],vocab_size,embed_dim=embedding_size,
            initializer=tf.random_uniform_initializer(-1.0,-1.0))
    
    # create an LSTM cell of size 100
#    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = 100)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = 50)
    # Multi 
#    num_units = [args.batch_size, 64]
    
 
    # Getting sequence length from features sucks -> initialize sequence length here
    sequence_length = tf.count_nonzero(features['x'], 1)
    
    # create the complete LSTM
    _, final_states = tf.nn.dynamic_rnn(
        lstm_cell, inputs, sequence_length = sequence_length, dtype=tf.float32)
    
    # get the final hidden states of dimensionality [batch_size x sentence_size]  [batch_size, lstm_units]
    # the last state for each element in the batch is final_states.h
    outputs = final_states.h   
    
    # Fully Connected Layer
    num_unit = 1
    out_points = tf.layers.dense(inputs=outputs, units=num_unit, name = "predicted_points")
    
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])
    
    # Compute loss.
#    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)    
#    loss = tf.losses.softmax_cross_entropy(labels,logits=logits)   
#    loss = tf.losses.mean_squared_error(labels,logits)
    
    # Compute predictions.
#    predictions = {
#      # Generate predictions (for PREDICT and EVAL mode)
#      "next": tf.round(tf.sigmoid(logits)),
#      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#      "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")
#      }

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "point": out_points
      }
    
    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions["point"])
    
    # Compute loss => NEED TO PUT THIS AFTER PREDICTION
#    loss = tf.losses.sigmoid_cross_entropy(labels,logits)    
    
    loss = tf.losses.mean_squared_error(labels,out_points)
    
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    labels = tf.cast(labels, tf.float32)
    
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "MSE": tf.metrics.mean_squared_error(
                    labels=labels, predictions=predictions["point"]),
            "MeanAbsoluteError": tf.metrics.mean_absolute_error(labels=labels, predictions=predictions["point"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops) 
    
    
def serving_input_receiver_fn():
    """
        Description: This is used to define inputs to serve the model.
        Usage:
        return: ServingInputReciever
        Ref: https://www.tensorflow.org/versions/r1.7/api_docs/python/tf/estimator/export/ServingInputReceiver
    """
    
    reciever_tensors = {
        # The size of input sentence is flexible.
        "sentence":tf.placeholder(tf.int32, [None,])
    }
    
    
    features = {
        # Resize given images.
        "x": tf.reshape(reciever_tensors["sentence"], [1,sentence_size])
    }
    
    
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)
    

def parser(x, y):
    '''
        Description: 
        Usage:
    '''
    
    features = {"x": x}
    return features, y


def train_input_fn(x_train,y_train,batch_size):
    '''
        Description: 
        Usage:
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    dataset = dataset.shuffle(1000).batch(batch_size).map(parser).repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn(x_train,y_train,batch_size):
    '''
        Description: 
        Usage:
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    # Don't shuffle when evaluation
    dataset = dataset.batch(batch_size).map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()  



def main(unused_argv):
    
    # Create the Estimator
    RNN_classifier = tf.estimator.Estimator(
        model_fn=LSTM_model_fn, model_dir= args.model_dir)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
#    tensors_to_log = {"predicted_points": "predicted_points"}
#    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        
#    RNN_classifier.train(
#        input_fn=lambda: train_input_fn(x_train,y_train,batch_size=args.batch_size),
#        steps=args.step_size,
#        hooks=[logging_hook])  

    RNN_classifier.train(
        input_fn=lambda: train_input_fn(x_train,y_train,batch_size=args.batch_size),
        steps=args.step_size)  
    
    eval_results = RNN_classifier.evaluate(
       input_fn = lambda: eval_input_fn(x_test,y_test,batch_size=args.batch_size))
    
    print(eval_results)
    
    # Save the model
    RNN_classifier.export_savedmodel(args.saved_dir, serving_input_receiver_fn=serving_input_receiver_fn)

if __name__ == "__main__":
    tf.app.run()
