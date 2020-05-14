'''
Sample Architecture using the Model API
'''
# typical useful libararies
from __future__ import print_function
import numpy as np
import os
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.preprocessing import sequence
from keras.models import Model
from keras.models import Sequential
import keras.layers
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Input, GlobalMaxPooling1D
from keras.datasets import imdb, reuters
from keras import backend as K
from keras.callbacks import Callback
import pdb

#Optional definitions 
BASE_DIR = 'YOUR BASE DIRECTORY'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
batch_size = 128
hidden_units = 150

#May setup directory for results--Optional
results_folder = 'Directory for your results'
if os.path.exists(results_folder) == False:
    os.makedirs(results_folder)
filename = os.path.join(results_folder, 'results.txt')

#Assuming your data is split between training and validation pairs
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

#DEFINE YOUR ARCHITECTURE using the MODEL API in KERAS

#Input layer
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))

#FOR EXAMPLE, LSTM3 processing applied to text embedding layer- model
x1 = Embedding(MAX_NB_WORDS, 300, input_length=MAX_SEQUENCE_LENGTH)(sequence_input)
lstm =  LSTM3(implementation= 1, units=hidden_units,
                    activation='sigmoid',
                    input_shape=x_train.shape[1:])
x1 = lstm(x1)

# ConvNet processing--model
embedded_sequences = embedding_layer(sequence_input)
x2 = Conv1D(150, 3, activation='relu')(embedded_sequences)
x2 = MaxPooling1D(3)(x2)
x2 = Conv1D(150, 3, activation='relu')(x2)
x2 = MaxPooling1D(3)(x2)
x2 = Conv1D(150, 3, activation='relu')(x2)
x2 = GlobalMaxPooling1D()(x2)
x2 = Dense(150, activation='relu')(x2)

# concatenate lstm-type and ConvNet models
conc = keras.layers.Concatenate()([x1, x2])
print(conc.shape)

#may perform furhter processing
x = Dropout(0.5)(conc)
preds = Dense(len(labels_index), activation='softmax')(x)

#Your final input-output model
model = Model(sequence_input, preds)

# You may try using different optimizers and different optimizer configs, e.g., 
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
## You may define a callback function to collect history, e.g. loss, ... 
history = callback_function(filename)
## YOU need to define the above function

#training using fit

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=100,
          callbacks=[history],
          validation_data=[x_val, y_val])