
# script for modelling dexcom alert data

# first load libraries
import pydub
from pydub import AudioSegment
from pydub.playback import play
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import scipy.io.wavfile
from scipy.io.wavfile import read
from scipy import signal
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from keras.regularizers import l2

# set random state for reproducibility in python, numpy and tf
tf.keras.utils.set_random_seed(123)

# read in dexcom alert recordings and segment
clear_alerts = AudioSegment.from_file('data/alerts/clear_alerts.m4a')
unclear_alerts = AudioSegment.from_file('data/alerts/under_blanket_alerts.m4a')
clear_high = clear_alerts[1000:4000]
clear_low = clear_alerts[7650:9650]
clear_urgent_low = clear_alerts[14500:17000]
unclear_high = unclear_alerts[700:3700]
unclear_low = unclear_alerts[6000:8000]
unclear_urgent_low = unclear_alerts[11500:14000]

# functions

# divide each of train, dev and test into eights (by alert type),
# two parts negative examples and one part from each of the six alert
# types: high, unclear high, low, unclear low, urgent low and unclear
# urgent low
def split_into_eights(X):
    split1, split2 = train_test_split(X, train_size = 0.5, random_state = 123)
    split11, split12 = train_test_split(split1, train_size = 0.5, random_state = 123)
    split111, split112 = train_test_split(split11, train_size = 0.5, random_state = 123)
    split121, split122 = train_test_split(split12, train_size = 0.5, random_state = 123)
    split21, split22 = train_test_split(split2, train_size = 0.5, random_state = 123)
    split211, split212 = train_test_split(split21, train_size = 0.5, random_state = 123)
    split221, split222 = train_test_split(split22, train_size = 0.5, random_state = 123)
    negative = split111 + split112
    high = split121
    low = split122
    urgent_low = split211
    unclear_high = split212
    unclear_low = split221
    unclear_urgent_low = split222
    return negative, high, low, urgent_low, unclear_high, unclear_low, unclear_urgent_low

# function to compute spectrogram from audio file, 
# potentially overlaying alert sound
def spectrogram(fname, alert = None):
    #  and set frame rate to frame rate of
    # dexcom alert recordings
    temp = AudioSegment.from_file(fname) # read in file
    temp = temp.set_frame_rate(48000) # set frame rate to frame rate of dexcom alert recordings
    if alert != None: # overlay alert over audio clip
        start_time = random.sample(range(len(temp) - len(globals()[alert])),k = 1) # randomly sample start time of alert
        temp = temp.overlay(globals()[alert], position = start_time[0])
    fname = "data/ESC-50-master/audio/temp.wav" # save temporary file to read back in with scipy
    temp.export(fname,format = "wav")
    sr_value, x_value = scipy.io.wavfile.read(fname)
    _, _, Sxx= signal.spectrogram(x_value,sr_value) # compute spectrogram with scipy
    Sxx = Sxx.swapaxes(0,1) # format to have correct axes and dimensions
    Sxx = np.expand_dims(Sxx, axis = 0)
    os.remove(fname) # remove temp file
    return Sxx

# function to create training, dev and test datasets
def generate_data():
    # splitting data into train, test and dev sets
    fnames = os.listdir('data/ESC-50-master/audio')
    train, subset = train_test_split(fnames, test_size = 0.2, random_state = 123)
    dev, test = train_test_split(subset, test_size = 0.5, random_state = 123)
    del subset
    train_negative, train_high, train_low, train_urgent_low, train_unclear_high, train_unclear_low, train_unclear_urgent_low = split_into_eights(train)
    dev_negative, dev_high, dev_low, dev_urgent_low, dev_unclear_high, dev_unclear_low, dev_unclear_urgent_low = split_into_eights(dev)
    test_negative, test_high, test_low, test_urgent_low, test_unclear_high, test_unclear_low, test_unclear_urgent_low = split_into_eights(test)
    # overlay correct alerts over each audio clip, convert to spectrograms
    # and concatenate
    train_negative = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = None) for f in train_negative],axis = 0)
    train_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low') for f in train_low],axis = 0)
    train_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high') for f in train_high],axis = 0)
    train_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low') for f in train_urgent_low],axis = 0)
    train_unclear_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low') for f in train_unclear_low],axis = 0)
    train_unclear_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high') for f in train_unclear_high],axis = 0)
    train_unclear_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low') for f in train_unclear_urgent_low],axis = 0)
    dev_negative = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = None) for f in dev_negative],axis = 0)
    dev_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low') for f in dev_low],axis = 0)
    dev_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high') for f in dev_high],axis = 0)
    dev_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low') for f in dev_urgent_low],axis = 0)
    dev_unclear_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low') for f in dev_unclear_low],axis = 0)
    dev_unclear_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high') for f in dev_unclear_high],axis = 0)
    dev_unclear_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low') for f in dev_unclear_urgent_low],axis = 0)
    test_negative = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = None) for f in test_negative],axis = 0)
    test_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low') for f in test_low],axis = 0)
    test_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high') for f in test_high],axis = 0)
    test_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low') for f in test_urgent_low],axis = 0)
    test_unclear_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low') for f in test_unclear_low],axis = 0)
    test_unclear_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high') for f in test_unclear_high],axis = 0)
    test_unclear_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low') for f in test_unclear_urgent_low],axis = 0)
    # combine into train, test and dev sets for features and labels
    X_train = np.concatenate([train_negative, train_high, train_unclear_high, train_low, train_unclear_low, train_urgent_low, train_unclear_urgent_low], axis = 0)
    X_dev = np.concatenate([dev_negative, dev_high, dev_unclear_high, dev_low, dev_unclear_low, dev_urgent_low, dev_unclear_urgent_low], axis = 0)
    X_test = np.concatenate([test_negative, test_high, test_unclear_high, test_low, test_unclear_low, test_urgent_low, test_unclear_urgent_low], axis = 0)
    del train_negative, train_high, train_unclear_high, train_low, train_unclear_low, train_urgent_low, train_unclear_urgent_low
    del dev_negative, dev_high, dev_unclear_high, dev_low, dev_unclear_low, dev_urgent_low, dev_unclear_urgent_low
    del test_negative, test_high, test_unclear_high, test_low, test_unclear_low, test_urgent_low, test_unclear_urgent_low
    # labels are one-hot encoded vectors from 4 classes
    negative = np.array([1,0,0,0]).reshape((1,4))
    high = np.array([0,1,0,0]).reshape((1,4))
    low = np.array([0,0,1,0]).reshape((1,4))
    urgent_low = np.array([0,0,0,1]).reshape((1,4))
    Y_train = np.concatenate([negative for x in range(400)] + [high for x in range(400)] + [low for x in range(400)] + [urgent_low for x in range(400)],axis = 0)
    Y_dev = np.concatenate([negative for x in range(50)] + [high for x in range(50)] + [low for x in range(50)] + [urgent_low for x in range(50)],axis = 0)
    Y_test = np.concatenate([negative for x in range(50)] + [high for x in range(50)] + [low for x in range(50)] + [urgent_low for x in range(50)],axis = 0)
    # split into batches for training speed
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(64)
    return train_dataset, dev_dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test

# generate data and save
train_dataset, dev_dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test = generate_data()
np.savetxt('data/modelling/train.txt', X_train.reshape(X_train.shape[0], -1))
np.savetxt('data/modelling/dev.txt', X_dev.reshape(X_dev.shape[0], -1))
np.savetxt('data/modelling/test.txt', X_test.reshape(X_test.shape[0], -1))

# current best model
def bi512_2D(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

bi_2D = bi512_2D((1071, 129))
bi_2D.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi_2D.summary() # 2761348 parameters
history = bi_2D.fit(train_dataset, epochs = 40, validation_data = dev_dataset)
# get test set accuracy
m = tf.keras.metrics.Accuracy()
m.update_state(np.argmax(Y_test, axis = 1),np.argmax(bi_2D.predict(X_test), axis = 1))
m.result().numpy() # 73.5% performance
bi_2D.save('analysis/model.keras')

# let's start with a recurrent neural network model with one GRU layer followed by a dense layer
def recurrent_model(input_shape):
    # get input
    input_spec = tf.keras.Input(input_shape)
    # CONV1D layer with 196 filters, a filter size of 15, and stride of 4, 
    # then BatchNorm and ReLu
    #X = tfl.Conv1D(filters=196,kernel_size=15,strides=4)(input_spec)
    #X = tfl.BatchNormalization()(X)
    #X = tfl.Activation('relu')(X)
    # Dropout with rate 0.8
    #X = tfl.Dropout(rate=0.8)(X)
    # First LSTM layer with 128 units with 0.8-rate dropout.
    X = tfl.GRU(units=128, return_sequences = False,dropout = 0.8)(input_spec) # final one has to have return_sequences = False
    # Batch norm
    #X = tfl.BatchNormalization()(X)
    # Second LSTM layer and batch norm
    #X = tfl.LSTM(units=128, return_sequences = True,dropout = 0.8)(X)
    #X = tfl.BatchNormalization()(X)
    # Dense layer
    outputs = tfl.Dense(4, activation = "softmax")(X)
    # output
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

# create model and print summary
rnn_model = recurrent_model((1071, 129))
rnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
rnn_model.summary() # 99972 trainable parameters

# train model
history = rnn_model.fit(train_dataset, epochs = 20, validation_data = dev_dataset)

# print final training and dev set accuracies and dev set confusion matrix
history.history['accuracy'][19] # about 22%
history.history['val_accuracy'][19] # about 25%, so about chance
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(rnn_model(X_dev), axis = 1)) # rows are real labels, columns are predicted labels

# need higher training set accuracy! Let's start by adding a second GRU layer
def recurrent_model_2layer(input_shape):
    # get input
    input_spec = tf.keras.Input(input_shape)
    # CONV1D layer with 196 filters, a filter size of 15, and stride of 4, 
    # then BatchNorm and ReLu
    #X = tfl.Conv1D(filters=196,kernel_size=15,strides=4)(input_spec)
    #X = tfl.BatchNormalization()(X)
    #X = tfl.Activation('relu')(X)
    # Dropout with rate 0.8
    #X = tfl.Dropout(rate=0.8)(X)
    # First LSTM layer with 128 units with 0.8-rate dropout.
    X = tfl.GRU(units=128, return_sequences = True,dropout = 0.8)(input_spec)
    X = tfl.GRU(units=128, return_sequences = False,dropout = 0.8)(X) # final one has to have return_sequences = False
    # Batch norm
    #X = tfl.BatchNormalization()(X)
    # Second LSTM layer and batch norm
    #X = tfl.LSTM(units=128, return_sequences = True,dropout = 0.8)(X)
    #X = tfl.BatchNormalization()(X)
    # Dense layer
    outputs = tfl.Dense(4, activation = "softmax")(X)
    # output
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

# create model and print summary
rnn_model_2layer = recurrent_model_2layer((1071, 129))
rnn_model_2layer.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
rnn_model_2layer.summary() # 199044 trainable parameters

# train model
history = rnn_model_2layer.fit(train_dataset, epochs = 20, validation_data = dev_dataset)

# print final training and dev set accuracies and dev set confusion matrix
history.history['accuracy'][19] # about 20%
history.history['val_accuracy'][19] # about 20%, so worse than chance..
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(rnn_model_2layer(X_dev), axis = 1)) # rows are real labels, columns are predicted labels
# mainly predicting highs and lows, but not good

# complex model
def complex_model(input_shape):
    # get input
    input_spec = tf.keras.Input(input_shape)
    # CONV1D layer with 196 filters, a filter size of 15, and stride of 4, 
    # then BatchNorm and ReLu
    X = tfl.Conv1D(filters=196,kernel_size=15,strides=4)(input_spec)
    X = tfl.BatchNormalization()(X)
    X = tfl.Activation('relu')(X)
    # Dropout with rate 0.8
    X = tfl.Dropout(rate=0.8)(X)
    # First LSTM layer with 128 units with 0.8-rate dropout.
    X = tfl.LSTM(units=128, return_sequences = True,dropout = 0.8)(X)
    # Batch norm
    X = tfl.BatchNormalization()(X)
    # Second LSTM layer and batch norm
    X = tfl.LSTM(units=128, return_sequences = False,dropout = 0.8)(X)
    X = tfl.BatchNormalization()(X)
    # Dense layer
    outputs = tfl.Dense(4, activation = "softmax")(X)
    # output
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

# create model and print summary
c_model = complex_model((1071, 129))
c_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
c_model.summary() # 678860 trainable parameters

# train model
history = c_model.fit(train_dataset, epochs = 20, validation_data = dev_dataset)

# print final training and dev set accuracies and dev set confusion matrix
history.history['accuracy'][19] # about 26%
history.history['val_accuracy'][19] # about 26%
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(c_model(X_dev), axis = 1)) # rows are real labels, columns are predicted labels
# mainly lows..

# trying a bidirectional LSTM model
def bi_rnn_model(input_shape):
    input_spec = tf.keras.Input(shape=input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 128,return_sequences = False))(input_spec)
    outputs = tfl.Dense(4,activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec,outputs = outputs)
    return model

bi_rnn = bi_rnn_model((1071, 129))
bi_rnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi_rnn.summary() # 265220 trainable parameters

history = bi_rnn.fit(train_dataset, epochs = 20, validation_data = dev_dataset)
history.history['accuracy'][19] # about 58%!
history.history['val_accuracy'][19] # about 29%
tf.math.confusion_matrix(labels = np.argmax(Y_train, axis = 1),predictions = np.argmax(bi_rnn(X_train), axis = 1)) # pretty great!
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(bi_rnn(X_dev), axis = 1)) # pretty uniform

# now let's try a complex bidirectional LSTM model
def bi_complex_rnn(input_shape):
    input_spec = tf.keras.Input(shape=input_shape)
    # CONV1D layer with 196 filters, a filter size of 15, and stride of 4, 
    # then BatchNorm and ReLu
    X = tfl.Conv1D(filters = 196,kernel_size = 15,strides = 4)(input_spec)
    X = tfl.BatchNormalization()(X)
    X = tfl.Activation('relu')(X)
    # Dropout with rate 0.8
    X = tfl.Dropout(rate = 0.8)(X)
    # First LSTM layer with 128 units with 0.8-rate dropout.
    X = tfl.Bidirectional(tfl.LSTM(units = 128,dropout = 0.8,return_sequences = True))(X)
    # Batch norm
    X = tfl.BatchNormalization()(X)
    # Second LSTM layer and batch norm
    X = tfl.Bidirectional(tfl.LSTM(units = 128,dropout = 0.8,return_sequences = False))(X)
    # Dense layer
    outputs = tfl.Dense(4,activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec,outputs = outputs)
    return model

bi_rnn = bi_complex_rnn((1071, 129))
bi_rnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi_rnn.summary() # 1108428 trainable params

history = bi_rnn.fit(train_dataset, epochs = 20, validation_data = dev_dataset)
# top training accuracy of about 64%, validation accuracy of 28%
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(bi_rnn(X_dev), axis = 1)) # only predicing lows and highs basically

# three layer simple bidirectional LSTM model
def bi_3layer(input_shape):
    input_spec = tf.keras.Input(shape=input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 128,return_sequences = True))(input_spec)
    X = tfl.Bidirectional(tfl.LSTM(units = 128,return_sequences = True))(X)
    X = tfl.Bidirectional(tfl.LSTM(units = 128,return_sequences = False))(X)
    outputs = tfl.Dense(4,activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec,outputs = outputs)
    return model

bi_3 = bi_3layer((1071, 129))
bi_3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi_3.summary() # 1053700 trainable parameters

history = bi_3.fit(train_dataset, epochs = 20, validation_data = dev_dataset) # not great

# now we iterate from the bidirectional simple RNN
def bi_rnn2(input_shape):
    input_spec = tf.keras.Input(shape=input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 128,return_sequences = False))(input_spec)
    X = tfl.Dropout(rate = 0.8)(X)
    X = tfl.BatchNormalization()(X)
    outputs = tfl.Dense(4,activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec,outputs = outputs)
    return model

bi2 = bi_rnn2((1071, 129))
bi2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi2.summary() # 265732 trainable params
history = bi2.fit(train_dataset, epochs = 20, validation_data = dev_dataset) # not great..

def bi_rnn_256(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 256,return_sequences = False))(input_spec)
    output = tfl.Dense(4,activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec,outputs = output)
    return model

bi_256 = bi_rnn_256((1071,129))
bi_256.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi_256.summary() # 792580 parameters
history = bi_256.fit(train_dataset, epochs = 20, validation_data = dev_dataset) # getting better...

def bi_rnn_512(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512,return_sequences = False))(input_spec)
    output = tfl.Dense(4,activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec,outputs = output)
    return model

bi512 = bi_rnn_512((1071, 129))
bi512.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi512.summary() # 2633732 params
history = bi512.fit(train_dataset, epochs = 40, validation_data = dev_dataset) # best yet by far!!
# final training accuracy 83%, dev accuracy 72%!!
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(bi512(X_dev), axis = 1)) # best at predicting urgent lows then no alerts then lows then highs.
# pretty sensitive and specific!

# now with dropout to deal with overfitting
def bi512_dropout(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, dropout = 0.8))(input_spec)
    output = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec,outputs = output)
    return model

bi512_drop = bi512_dropout((1071, 129))
bi512_drop.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi512_drop.summary() # same number of parameters as bi512
history = bi512_drop.fit(train_dataset, epochs = 40, validation_data = dev_dataset) # not great..

# now let's try with more hidden units
def bi_rnn_1024(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 1024,return_sequences = False))(input_spec)
    outputs = tfl.Dense(4,activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec,outputs = outputs)
    return model

bi1024 = bi_rnn_1024((1071,129))
bi1024.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
bi1024.summary() # 9461764 trainable params (wow..)
history = bi1024.fit(train_dataset, epochs = 20, validation_data = dev_dataset) # about the same as the 512 model..

# circling back to the bidirectional model with 512 hidden units:
history = bi512.fit(train_dataset, epochs = 40, validation_data = dev_dataset)

# new model with L2 regularization parameter 1
def regularized_1(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, kernel_regularizer = l2(1e-5)))(input_spec)
    X = tfl.Dense(128, activation = 'tanh', kernel_regularizer = l2(1e-5))(X)
    outputs = tfl.Dense(4, activation = 'softmax', kernel_regularizer = l2(1e-5))(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

r1 = regularized_1((1071, 129))
r1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
r1.summary()
history = r1.fit(train_dataset, epochs = 40, validation_data = dev_dataset)

# new model with L2 regularization parameter 1e-3
def regularized_2(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, kernel_regularizer = l2(1e-3)))(input_spec)
    X = tfl.Dense(128, activation = 'tanh', kernel_regularizer = l2(1e-3))(X)
    outputs = tfl.Dense(4, activation = 'softmax', kernel_regularizer = l2(1e-3))(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

r2 = regularized_2((1071, 129))
r2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
r2.summary()
history = r2.fit(train_dataset, epochs = 40, validation_data = dev_dataset)

# new model with L2 regularization parameter 1e-1
def regularized_3(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, kernel_regularizer = l2(1e-1)))(input_spec)
    X = tfl.Dense(128, activation = 'tanh', kernel_regularizer = l2(1e-1))(X)
    outputs = tfl.Dense(4, activation = 'softmax', kernel_regularizer = l2(1e-1))(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

r3 = regularized_3((1071, 129))
r3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
r3.summary()
history = r3.fit(train_dataset, epochs = 40, validation_data = dev_dataset)

# now let's try a convolutional model
def convolutional_model(input_shape):
    '''
    A simple convolutional model with two CONV2D->RELU->MAXPOOL blocks,
    followed by a dense layer.
    '''
    # get input
    input_spec = tf.keras.Input(shape=input_shape)
    # CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    X = tfl.Conv2D(filters = 8,kernel_size = 4,padding = 'same')(input_spec)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (8,8),strides = (8,8),padding = 'same')(X)
    # CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    X = tfl.Conv2D(filters = 8,kernel_size = 4,padding = 'same')(X)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (8,8),strides = (8,8),padding = 'same')(X)
    # FLATTEN
    X = tfl.Flatten()(X)
    # Dense layer
    # 4 neurons in output layer.
    outputs = tfl.Dense(units = 4,activation = 'softmax')(X)
    model = tf.keras.Model(inputs=input_spec, outputs=outputs)
    return model

# compile model and summarize
conv_model = convolutional_model((1071, 129, 1)) # need extra dimension for "gray scale"
conv_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
conv_model.summary() # 2804 (trainable) params

# train model
history = conv_model.fit(train_dataset, epochs=20, validation_data=dev_dataset)

# print final training and dev set accuracies and dev set confusion matrix
history.history['accuracy'][19] # about 28%
history.history['val_accuracy'][19] # about 30%, so not great..
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(conv_model(X_dev), axis = 1)) # rows are real labels, columns are predicted labels
# low was predicted a lot, out of real low labels it performed well (sensitive) but otherwise not (not specific)
# lows are pretty easy - clear low frequency same note tones
