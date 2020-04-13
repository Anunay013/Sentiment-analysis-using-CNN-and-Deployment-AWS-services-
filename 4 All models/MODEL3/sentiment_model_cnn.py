"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf
import numpy as np
import boto3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling
    """
    # load the whole embedding into memory
    embeddings_index = dict()

    if "s3://" in config["embeddings_path"]:
        print('Fetching embeddings from S3...')
        s3 = boto3.resource('s3')
        path = config["embeddings_path"]
        path = path[5:].split("/")

        bucketname = path[0]
        filename = "/".join(path[1:]) 
        s3.Bucket(bucketname).download_file(filename, path[-1])

        embeddings_index = dict()
        f = open(path[-1], encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

    else:
        f = open(config["embeddings_path"], encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))

    vocab_size = config["embeddings_dictionary_size"]

    n = len(embeddings_index.keys())
    m = len(embeddings_index['the'])

    embedding_matrix = np.zeros((n,m))
    for index, key in zip(range(0, n), embeddings_index.keys()):
        embedding_matrix[index] = embeddings_index[key]

    # define model
    model = Sequential()

    model.add(Embedding(vocab_size, config["embeddings_vector_size"], weights=[embedding_matrix], input_length=config["padding_size"], trainable=True, name='embedding'))

    model.add(Conv1D(filters = 128, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters = 256, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters = 512, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters = 1024, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu'))
    model.add(MaxPooling1D(2))

    model.add(Flatten())

    model.add(Dense(1024, activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    adam = Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    print('Defined model')
    
    return model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """

    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))

