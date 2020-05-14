#%%
"""
(Now using slim21 module, which is compatible with Keras2.1--backend: Tensorflow)

Sample script for importing the SLIM21 models and executing multiple training/
testing. 
Results are saved/dumpped in binary files in a folder. 
Then, one uses the 
DRAW
script to plot the results

CSANN LAB--MSU--msu.edu
Contributors:
Atra Akandeh
Fathi Salem
...

"""
from __future__ import print_function
from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
#from keras.layers import LSTM
import numpy as np
## The following is used only ofr Theano backend
##from theano.tensor.shared_randomstreams import RandomStreams
##
import pickle
import os.path
#
##Replace the following path with your own folders/directory where 
##the module slim21.py is located
pth = 'C:/Users/salem.ECE451/KERAS2/keras21/New'
pth1 = 'C:/Users/salem.ECE451/KERAS2/keras21/New/Results/'
os.chdir(pth)
##
from slim21 import LSTMs
#

#np.random.seed(3)
#srgn = RandomStreams(3)

batch_size = 32
nb_classes = 2
nb_epochs = 1
hidden_units = 400
embedding_vector_length = 32

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

## This is for comment%%

acts = ['sigmoid' , 'tanh']
#include all your models ina list
lstms = ['LSTM1'] #, 'LSTM2', 'LSTM3','LSTM4', 'LSTM5', 'LSTM6', 'LSTM4a', 'LSTM5a', 'LSTM10', 'LSTM11']
#inlcude all your lr (grid) in a list
etas = [1.2e-5]
#Use the name label of the model for the file
names = ['lstm1']#, 'lstm2', 'lstm3','lstm4', 'lstm5', 'lstm6', 'lstm4a', 'lstm5a', 'lstm10', 'lstm11']

for act in acts:
    for eta in etas:
        sub = '%s-eta%.4g' % (act,eta)
        
        if not os.path.exists(sub):
            os.mkdir(sub)
            print("Directory " , sub ,  " Created ")
        else:    
            print("Directory " , sub ,  " already exists")
            
        for cnt, lstm in enumerate(lstms):
            
            np.random.seed(3)
##May use iF using RandomStreams in Theano            
#            srgn = RandomStreams(3)
#

            model = Sequential()
            model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length, trainable=False))           
            lstmi = LSTMs(implementation= 1, units=hidden_units,
                    activation=act,
                    input_shape=X_train.shape[1:], model=lstm)
            model.add(lstmi)
            model.add(Dense(1, activation=act))
            adam = Adam(lr=eta, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
            model.summary()
            hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                             verbose=1, validation_data=(X_test, y_test))
            fn = '%s.p' % names[cnt]     
            final = os.path.join(pth + '/' +sub, fn)
            A = hist.history
            pickle.dump( A , open( final, "wb" ) )