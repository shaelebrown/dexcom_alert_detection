
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
import keras
from keras import backend as K

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
def spectrogram(fname, alert = None, index = 0, type = 'train'):
    #  and set frame rate to frame rate of
    # dexcom alert recordings
    temp = AudioSegment.from_file(fname) # read in file
    temp = temp.set_frame_rate(48000) # set frame rate to frame rate of dexcom alert recordings
    if alert != None: # overlay alert over audio clip
        start_time = random.sample(range(len(temp) - len(globals()[alert])),k = 1) # randomly sample start time of alert
        temp = temp.overlay(globals()[alert], position = start_time[0])
    fname = "data/modelling/modified_audio_files/" + type + '_' + str(index) + '.wav' # save new audio file to read back in with scipy
    temp.export(fname,format = "wav")
    sr_value, x_value = scipy.io.wavfile.read(fname)
    _, _, Sxx= signal.spectrogram(x_value,sr_value) # compute spectrogram with scipy
    Sxx = Sxx.swapaxes(0,1) # format to have correct axes and dimensions
    Sxx = np.expand_dims(Sxx, axis = 0)
    if type == 'gen':
        os.remove(fname)
    return Sxx

# function to create training, dev and test datasets
# this function is old
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
    train_negative = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = None, type = 'train', index = train_negative.index(f)) for f in train_negative],axis = 0)
    train_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low', type = 'train', index = train_low.index(f) + 800) for f in train_low],axis = 0)
    train_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high', type = 'train', index = train_high.index(f) + 400) for f in train_high],axis = 0)
    train_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low', type = 'train', index = train_urgent_low.index(f) + 1200) for f in train_urgent_low],axis = 0)
    train_unclear_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low', type = 'train', index = train_unclear_low.index(f) + 1000) for f in train_unclear_low],axis = 0)
    train_unclear_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high', type = 'train', index = train_unclear_high.index(f) + 600) for f in train_unclear_high],axis = 0)
    train_unclear_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low', type = 'train', index = train_unclear_urgent_low.index(f) + 1400) for f in train_unclear_urgent_low],axis = 0)
    dev_negative = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = None, type = 'dev', index = dev_negative.index(f)) for f in dev_negative],axis = 0)
    dev_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low', type = 'dev', index = dev_low.index(f) + 100) for f in dev_low],axis = 0)
    dev_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high', type = 'dev', index = dev_high.index(f) + 50) for f in dev_high],axis = 0)
    dev_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low', type = 'dev',index = dev_urgent_low.index(f) + 150) for f in dev_urgent_low],axis = 0)
    dev_unclear_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low', type = 'dev', index = dev_unclear_low.index(f) + 125) for f in dev_unclear_low],axis = 0)
    dev_unclear_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high', type = 'dev', index = dev_unclear_high.index(f) + 75) for f in dev_unclear_high],axis = 0)
    dev_unclear_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low', type = 'dev', index = dev_unclear_urgent_low.index(f) + 175) for f in dev_unclear_urgent_low],axis = 0)
    test_negative = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = None, type = 'test', index = test_negative.index(f)) for f in test_negative],axis = 0)
    test_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low', type = 'test', index = test_low.index(f) + 100) for f in test_low],axis = 0)
    test_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high', type = 'test', index = test_high.index(f) + 50) for f in test_high],axis = 0)
    test_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low', type = 'test', index = test_urgent_low.index(f) + 150) for f in test_urgent_low],axis = 0)
    test_unclear_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low', type = 'test', index = test_unclear_low.index(f) + 125) for f in test_unclear_low],axis = 0)
    test_unclear_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high', type = 'test', index = test_unclear_high.index(f) + 75) for f in test_unclear_high],axis = 0)
    test_unclear_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low', type = 'test', index = test_unclear_urgent_low.index(f) + 175) for f in test_unclear_urgent_low],axis = 0)
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
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(50)
    dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(50)
    return train_dataset, dev_dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test

# function to create larger training, dev and test datasets
def generate_larger_data():
    # splitting data into train, test and dev sets
    fnames = os.listdir('data/ESC-50-master/audio')
    # generate 7 data frames of spectrograms, one for each output type, then
    # split into train, dev and test sets
    clear_high_df = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high', type = 'gen', index = fnames.index(f)) for f in random.sample(fnames, k = 1000)],axis = 0)
    unclear_high_df = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high', type = 'gen', index = fnames.index(f)) for f in random.sample(fnames, k = 1000)],axis = 0)
    clear_low_df = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low', type = 'gen', index = fnames.index(f)) for f in random.sample(fnames, k = 1000)],axis = 0)
    unclear_low_df = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low', type = 'gen', index = fnames.index(f)) for f in random.sample(fnames, k = 1000)],axis = 0)
    clear_urgent_low_df = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low', type = 'gen', index = fnames.index(f)) for f in random.sample(fnames, k = 1000)],axis = 0)
    unclear_urgent_low_df = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low', type = 'gen', index = fnames.index(f)) for f in random.sample(fnames, k = 1000)],axis = 0)
    negative_df = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, type = 'gen', index = fnames.index(f)) for f in fnames],axis = 0)
    train_clear_high, subset_clear_high = train_test_split(clear_high_df, test_size = 0.2, random_state = 123)
    dev_clear_high, test_clear_high = train_test_split(subset_clear_high, test_size = 0.5, random_state = 123)
    del subset_clear_high
    train_unclear_high, subset_unclear_high = train_test_split(unclear_high_df, test_size = 0.2, random_state = 123)
    dev_unclear_high, test_unclear_high = train_test_split(subset_unclear_high, test_size = 0.5, random_state = 123)
    del subset_unclear_high
    train_clear_low, subset_clear_low = train_test_split(clear_low_df, test_size = 0.2, random_state = 123)
    dev_clear_low, test_clear_low = train_test_split(subset_clear_low, test_size = 0.5, random_state = 123)
    del subset_clear_low
    train_unclear_low, subset_unclear_low = train_test_split(unclear_low_df, test_size = 0.2, random_state = 123)
    dev_unclear_low, test_unclear_low = train_test_split(subset_unclear_low, test_size = 0.5, random_state = 123)
    del subset_unclear_low
    train_clear_urgent_low, subset_clear_urgent_low = train_test_split(clear_urgent_low_df, test_size = 0.2, random_state = 123)
    dev_clear_urgent_low, test_clear_urgent_low = train_test_split(subset_clear_urgent_low, test_size = 0.5, random_state = 123)
    del subset_clear_urgent_low
    train_unclear_urgent_low, subset_unclear_urgent_low = train_test_split(unclear_urgent_low_df, test_size = 0.2, random_state = 123)
    dev_unclear_urgent_low, test_unclear_urgent_low = train_test_split(subset_unclear_urgent_low, test_size = 0.5, random_state = 123)
    del subset_unclear_urgent_low
    train_negative, subset_negative = train_test_split(negative_df, test_size = 0.2, random_state = 123)
    dev_negative, test_negative= train_test_split(subset_negative, test_size = 0.5, random_state = 123)
    del subset_negative
    train_high = np.concatenate([train_clear_high, train_unclear_high], axis = 0)
    train_low = np.concatenate([train_clear_low, train_unclear_low], axis = 0)
    train_urgent_low = np.concatenate([train_clear_urgent_low, train_unclear_urgent_low], axis = 0)
    dev_high = np.concatenate([dev_clear_high, dev_unclear_high], axis = 0)
    dev_low = np.concatenate([dev_clear_low, dev_unclear_low], axis = 0)
    dev_urgent_low = np.concatenate([dev_clear_urgent_low, dev_unclear_urgent_low], axis = 0)
    test_high = np.concatenate([test_clear_high, test_unclear_high], axis = 0)
    test_low = np.concatenate([test_clear_low, test_unclear_low], axis = 0)
    test_urgent_low = np.concatenate([test_clear_urgent_low, test_unclear_urgent_low], axis = 0)
    # combine into train, test and dev sets for features and labels
    X_train = np.concatenate([train_negative, train_high, train_low, train_urgent_low], axis = 0)
    X_dev = np.concatenate([dev_negative, dev_high, dev_low, dev_urgent_low], axis = 0)
    X_test = np.concatenate([test_negative, test_high, test_low, test_urgent_low], axis = 0)
    del train_negative, train_high, train_unclear_high, train_low, train_unclear_low, train_urgent_low, train_unclear_urgent_low
    del dev_negative, dev_high, dev_unclear_high, dev_low, dev_unclear_low, dev_urgent_low, dev_unclear_urgent_low
    del test_negative, test_high, test_unclear_high, test_low, test_unclear_low, test_urgent_low, test_unclear_urgent_low
    # labels are one-hot encoded vectors from 4 classes
    negative = np.array([1,0,0,0]).reshape((1,4))
    high = np.array([0,1,0,0]).reshape((1,4))
    low = np.array([0,0,1,0]).reshape((1,4))
    urgent_low = np.array([0,0,0,1]).reshape((1,4))
    Y_train = np.concatenate([negative for x in range(1600)] + [high for x in range(1600)] + [low for x in range(1600)] + [urgent_low for x in range(1600)],axis = 0)
    Y_dev = np.concatenate([negative for x in range(200)] + [high for x in range(200)] + [low for x in range(200)] + [urgent_low for x in range(200)],axis = 0)
    Y_test = np.concatenate([negative for x in range(200)] + [high for x in range(200)] + [low for x in range(200)] + [urgent_low for x in range(200)],axis = 0)
    # split into batches for training speed
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(64)
    return train_dataset, dev_dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test

# generate data
train_dataset, dev_dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test = generate_data()

# current best model
def bi512_2D(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, dropout = 0.1))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

# read in old best model to fine-tune
model = keras.models.load_model('analysis/model.keras')
model.compile(optimizer = tf.keras.optimizers.AdamW(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_dataset, epochs = 20, validation_data = dev_dataset)

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
# first let's subset training and dev sets
X_train = X_train[:,:,range(11)]
X_dev = X_dev[:,:,range(11)]
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(64)
def convolutional_model(input_shape):
    # get input
    input_spec = tf.keras.Input(shape=input_shape)
    # CONV2D for low alert: 4 filters 9x321, stride of 1, padding 'SAME'
    X = tfl.Conv2D(filters = 4,kernel_size = (321, 1),padding = 'same')(input_spec)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(X)
    # CONV2D for high alert: 4 filters 9x321, stride of 1, padding 'SAME'
    Y = tfl.Conv2D(filters = 4,kernel_size = (321,1),padding = 'same')(input_spec)
    # RELU
    Y = tfl.ReLU()(Y)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    Y = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(Y)
    # CONV2D for urgent low alert: 4 filters 9x321, stride of 1, padding 'SAME'
    Z = tfl.Conv2D(filters = 4,kernel_size = (321,1),padding = 'same')(input_spec)
    # RELU
    Z = tfl.ReLU()(Z)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    Z = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(Z)
    # CONCATENATE
    CON = tfl.Concatenate()([X,Y,Z])
    # CONV2D
    CON = tfl.Conv2D(filters = 6,kernel_size = (63,1),padding = 'same')(CON)
    # FLATTEN
    CON = tfl.Flatten()(CON)
    # Dense layer
    # 4 neurons in output layer.
    outputs = tfl.Dense(units = 4,activation = 'softmax')(CON)
    model = tf.keras.Model(inputs=input_spec, outputs=outputs)
    return model

# compile model and summarize
conv_model = convolutional_model((1071, 11, 1)) # need extra dimension for "gray scale"
conv_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
conv_model.summary() # 12898 (trainable) params

# train model
history = conv_model.fit(train_dataset, epochs=20, validation_data=dev_dataset)

# deeper conv model
def deep_conv_model(input_shape):
    # get input
    input_spec = tf.keras.Input(shape=input_shape)
    # CONV2D
    X = tfl.Conv2D(filters = 12,kernel_size = (321, 1),padding = 'same')(input_spec)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(X)
    # CONV2D for low alert: 4 filters 9x321, stride of 1, padding 'SAME'
    X = tfl.Conv2D(filters = 12,kernel_size = (321, 1),padding = 'same')(X)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(X)
    # CONV2D for low alert: 4 filters 9x321, stride of 1, padding 'SAME'
    X = tfl.Conv2D(filters = 12,kernel_size = (321, 1),padding = 'same')(X)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(X)
    F = tfl.Flatten()(X)
    # Dense layers
    F = tfl.Dense(128,'tanh')(F)
    outputs = tfl.Dense(units = 4,activation = 'softmax')(F)
    model = tf.keras.Model(inputs=input_spec, outputs=outputs)
    return model

deep_cnn = deep_conv_model((1071, 11, 1))
deep_cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
deep_cnn.summary()
history = deep_cnn.fit(train_dataset, epochs=20, validation_data=dev_dataset)

# adding more FC layers
def deeper_conv_model(input_shape):
    # get input
    input_spec = tf.keras.Input(shape=input_shape)
    # CONV2D
    X = tfl.Conv2D(filters = 12,kernel_size = (321, 1),padding = 'same')(input_spec)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(X)
    # CONV2D for low alert: 4 filters 9x321, stride of 1, padding 'SAME'
    X = tfl.Conv2D(filters = 12,kernel_size = (321, 1),padding = 'same')(X)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(X)
    # CONV2D for low alert: 4 filters 9x321, stride of 1, padding 'SAME'
    X = tfl.Conv2D(filters = 12,kernel_size = (321, 1),padding = 'same')(X)
    # RELU
    X = tfl.ReLU()(X)
    # MAXPOOL: window 9x321, stride 1x35, padding 'SAME'
    X = tfl.MaxPool2D(pool_size = (321,1),strides = (63,1),padding = 'same')(X)
    F = tfl.Flatten()(X)
    # Dense layers
    F = tfl.Dense(128,'tanh')(F)
    F = tfl.Dense(64,'tanh')(F)
    F = tfl.Dense(32,'tanh')(F)
    F = tfl.Dense(16,'tanh')(F)
    F = tfl.Dense(8,'tanh')(F)
    outputs = tfl.Dense(units = 4,activation = 'softmax')(F)
    model = tf.keras.Model(inputs=input_spec, outputs=outputs)
    return model

deeper_cnn = deeper_conv_model((1071, 11, 1))
deeper_cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
deeper_cnn.summary()
history = deeper_cnn.fit(train_dataset, epochs=20, validation_data=dev_dataset)

# best model but on subsetted data
def bi512_2D_sub(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

bi_sub = bi512_2D_sub((1071, 11))
bi_sub.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
bi_sub.summary() # 2,278,020 params
history = bi_sub.fit(train_dataset, epochs = 20, validation_data = dev_dataset)

# how about with more dense layers
def bi512_2D_sub_deep(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    X = tfl.Dense(64, activation = 'tanh')(X)
    X = tfl.Dense(32, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

mod = bi512_2D_sub_deep((1071, 11))
mod.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
mod.summary() # 2,287,972 parameters

history = mod.fit(train_dataset, epochs = 20, validation_data = dev_dataset)

# current best model but with RSA loss function and multireturn
def RSA_512(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, dropout = 0.1))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = [X, outputs])
    return model

# read in old best model to fine-tune
model = RSA_512((1071, 129))

def correlation_distance(x, y_batch):
    epsilon = 1e-9
    y_batch = np.argmax(y_batch,axis = 1)
    y_tens_sq = np.zeros(shape = (len(y_batch),len(y_batch)))
    for i in range(len(y_batch)-1):
        for j in range(i+1,len(y_batch)):
            if y_batch[i] != y_batch[j]:
                y_tens_sq[i,j] = 1
                y_tens_sq[j,i] = 1
    y_tens_sq = tf.Variable(y_tens_sq,dtype = tf.float32)
    indices = tf.transpose(tf.Variable(np.triu_indices(y_tens_sq.shape[0], k = 1)))
    y = tf.gather_nd(y_tens_sq, indices = indices)
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return 1 - K.mean(r)

def distance_matrix(X):
    r = tf.reshape(tf.reduce_sum(X*X, 1),(1, tf.shape(X)[0].numpy()))
    r2 = tf.tile(r, [tf.shape(X)[0].numpy(), 1])
    D = tf.sqrt(tf.math.maximum(0.0,r2 - 2*tf.matmul(X, tf.transpose(X)) + tf.transpose(r2)))
    indices = tf.transpose(tf.Variable(np.triu_indices(D.shape[0], k = 1)))
    upper = tf.gather_nd(D, indices = indices)
    return upper

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
loss_fn = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()

def apply_gradient(optimizer, model, x, y, lambd):
  with tf.GradientTape() as tape:
    activations, logits = model(x)
    D = distance_matrix(activations)
    entropy_term = lambd*loss_fn(y_true=y, y_pred=logits)
    RSA_term = (1-lambd)*correlation_distance(D, y)*np.ones((x.shape[0],))
    loss_value =  entropy_term + RSA_term
  
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  
  return logits, loss_value

def train_data_for_one_epoch():
  losses = []
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      logits, loss_value = apply_gradient(optimizer, model, x_batch_train, y_batch_train, lambd)
      losses.append(loss_value)
      train_acc_metric.update_state(y_batch_train, logits)
      tf.print('Finished batch')
  return losses

def perform_validation():
  losses = []
  for step, (x_batch_dev, y_batch_dev) in enumerate(dev_dataset):
        activations, logits = model(x_batch_dev)
        D = distance_matrix(activations)
        entropy_term = lambd*loss_fn(y_true=y_batch_dev, y_pred=logits)
        RSA_term = (1-lambd)*correlation_distance(D, y_batch_dev)*np.ones((x_batch_dev.shape[0],))
        loss_value =  entropy_term + RSA_term
        losses.append(loss_value)
        val_acc_metric.update_state(y_batch_dev, logits)
  return losses

# Iterate over epochs.
epochs = 40
epochs_val_losses, epochs_train_losses = [], []
lambd = 0.001
for epoch in range(epochs):
  print('Start of epoch %d' % (epoch,))
  losses_train = train_data_for_one_epoch()
  train_acc = train_acc_metric.result()
  losses_val = perform_validation()
  val_acc = val_acc_metric.result()
  losses_train_mean = np.mean(losses_train)
  losses_val_mean = np.mean(losses_val)
  epochs_val_losses.append(losses_val_mean)
  epochs_train_losses.append(losses_train_mean)
  print('\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f, Train Accuracy: %.4f, Validation Accuracy %.4f' % (epoch, float(losses_train_mean), float(losses_val_mean), float(train_acc), float(val_acc)))
  train_acc_metric.reset_states()
  val_acc_metric.reset_states()

# current best model
def best_model(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, dropout = 0.1))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

model = best_model((1071, 129))

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
loss_fn = tf.keras.losses.categorical_crossentropy
def weighted_loss_fun(y_true, y_pred):
   num_in_batch = y_pred.shape[0]
   relative_weights = np.array([1, 2, 3, 4]) # negative, high, low, urgent low
   counts = np.sum(y_true, 0)
   dot_prod = np.sum(counts*relative_weights)
   adjusted_weights = relative_weights/dot_prod
   sample_weights = adjusted_weights[np.argmax(y_true, axis = 1)]
   return loss_fn(y_true = y_true, y_pred = y_pred, sample_weight = sample_weights)
optimizer = tf.keras.optimizers.Adam()

def apply_gradient_pre20(optimizer, model, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss_value = loss_fn(y_true=y, y_pred=logits)
  
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  
  return logits, loss_value

def apply_gradient_post20(optimizer, model, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    reg_term0 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[0]))
    reg_term1 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[1]))
    reg_term2 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[2]))
    reg_term3 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[3]))
    reg_term4 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[4]))
    reg_term5 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[5]))
    reg_term6 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[6]))
    reg_term7 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[7]))
    reg_term8 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[8]))
    reg_term9 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[9]))
    reg_term = tf.math.add_n([reg_term0, reg_term1, reg_term2, reg_term3, reg_term4, reg_term5, reg_term6, reg_term7, reg_term8, reg_term9])
    entropy_term = loss_fn(y_true=y, y_pred=logits)
    loss_value = (1-lambd)*entropy_term + lambd*reg_term
  
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  
  return logits, loss_value

def train_data_for_one_epoch_pre20():
  losses = []
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      logits, loss_value = apply_gradient_pre20(optimizer, model, x_batch_train, y_batch_train)
      losses.append(loss_value)
      train_acc_metric.update_state(y_batch_train, logits)
  return losses

def train_data_for_one_epoch_post20():
  losses = []
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      logits, loss_value = apply_gradient_post20(optimizer, model, x_batch_train, y_batch_train)
      losses.append(loss_value)
      train_acc_metric.update_state(y_batch_train, logits)
  return losses

def perform_validation_pre20():
  losses = []
  for step, (x_batch_dev, y_batch_dev) in enumerate(dev_dataset):
        logits = model(x_batch_dev)
        loss_value = loss_fn(y_true=y_batch_dev, y_pred=logits)
        losses.append(loss_value)
        val_acc_metric.update_state(y_batch_dev, logits)
  return losses

def perform_validation_post20():
  losses = []
  for step, (x_batch_dev, y_batch_dev) in enumerate(dev_dataset):
        logits = model(x_batch_dev)
        reg_term0 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[0]))
        reg_term1 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[1]))
        reg_term2 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[2]))
        reg_term3 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[3]))
        reg_term4 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[4]))
        reg_term5 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[5]))
        reg_term6 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[6]))
        reg_term7 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[7]))
        reg_term8 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[8]))
        reg_term9 = tf.math.reduce_sum(tf.math.square(model.trainable_weights[9]))
        reg_term = tf.math.add_n([reg_term0, reg_term1, reg_term2, reg_term3, reg_term4, reg_term5, reg_term6, reg_term7, reg_term8, reg_term9])
        entropy_term = loss_fn(y_true=y_batch_dev, y_pred=logits)
        loss_value = (1-lambd)*entropy_term + lambd*reg_term
        losses.append(loss_value)
        val_acc_metric.update_state(y_batch_dev, logits)
  return losses

# Iterate over epochs.
# After epoch 20, add regularization
# After epoch 30, decrease learning rate
epochs = 40
epochs_val_losses, epochs_train_losses = [], []
lambd = 1000/2*(2761348 ** 2)
for epoch in range(epochs):
  print('Start of epoch %d' % (epoch,))
  if epoch < 20:
    losses_train = train_data_for_one_epoch_pre20()
  else:
    if epoch == 20 or epoch == 30:
      old_lr = optimizer.lr.read_value()
      new_lr = 0.1*old_lr
      optimizer.lr.assign(new_lr)
    losses_train = train_data_for_one_epoch_post20()
  train_acc = train_acc_metric.result()
  if epoch < 20:
     losses_val = perform_validation_pre20()
  else:
     losses_val = perform_validation_post20()
  val_acc = val_acc_metric.result()
  losses_train_mean = np.mean(losses_train)
  losses_val_mean = np.mean(losses_val)
  epochs_val_losses.append(losses_val_mean)
  epochs_train_losses.append(losses_train_mean)
  print('\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f, Train Accuracy: %.4f, Validation Accuracy %.4f' % (epoch, float(losses_train_mean), float(losses_val_mean), float(train_acc), float(val_acc)))
  train_acc_metric.reset_states()
  val_acc_metric.reset_states()

# current best model
def best_model(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, dropout = 0.1))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model

model = best_model((1071, 129))

loss_fn = tf.keras.losses.CategoricalCrossentropy()
def weighted_loss_fun(y_true, y_pred):
   num_in_batch = y_pred.shape[0]
   relative_weights = tf.constant([1, 2, 3, 4], dtype = tf.int64, shape = (1, 4)) # negative, high, low, urgent low
   counts = tf.reduce_sum(y_true, 0)
   dot_prod = tf.reduce_sum(counts*relative_weights)
   sample_weight = tf.matmul(y_true, tf.transpose(relative_weights))/dot_prod
   return loss_fn(y_true = y_true, y_pred = y_pred, sample_weight = sample_weight)
model.compile(optimizer = 'adam', loss = weighted_loss_fun, metrics = ['accuracy'])
history = model.fit(train_dataset, epochs = 20, validation_data = dev_dataset)

# trying again with conv models on subsetted spectrograms
X_train = X_train[:,:,range(16)]
X_dev = X_dev[:,:,range(16)]
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(50)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(50)
def conv(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 8,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Conv2D(filters = 16,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Conv2D(filters = 32,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
conv_model = conv((1071,16,1))
conv_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
conv_model.summary()
history = conv_model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# 60% train accuracy, 45% dev accuracy

def conv_deeper(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 8,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Conv2D(filters = 16,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Conv2D(filters = 32,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.Flatten()(X)
   X = tfl.Dense(32,activation = 'tanh')(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
conv_model = conv_deeper((1071,16,1))
conv_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
conv_model.summary()
history = conv_model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# worse

def conv_more_filters(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 16,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Conv2D(filters = 32,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Conv2D(filters = 64,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
conv_model = conv_more_filters((1071,16,1))
conv_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
conv_model.summary()
history = conv_model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# worse

def conv_bn(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 8,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 16,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 32,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model1 = conv_bn((1071,16,1))
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model1.summary()
history = model1.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# best!! 100% train accuracy and 65% dev accuracy

def conv_bn_drop(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 8,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 16,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 32,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   X = tfl.Dropout(rate = 0.1)(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model2 = conv_bn_drop((1071,16,1))
model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model2.summary()
history = model2.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# best!! 100% train accuracy, 70% dev accuracy

def conv_bn_more_drop(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 8,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Dropout(rate = 0.1)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 16,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Dropout(rate = 0.1)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 32,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.Dropout(rate = 0.1)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model3 = conv_bn_more_drop((1071,16,1))
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model3.summary()
history = model3.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# best!! 100% train accuracy, 70% dev accuracy

# now let's try with L2 regularization
def conv_bn(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 8,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 16,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 32,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
conv_model = conv_bn((1071,16,1))
vars = model.trainable_weights
@tf.autograph.experimental.do_not_convert
def loss_fun(lambd):
   cat_crossent = tf.keras.losses.CategoricalCrossentropy()
   def l(y_true, y_pred):
      CE = cat_crossent(y_true, y_pred)
      L2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * lambd
      return CE + L2
   return l
conv_model.compile(optimizer='adam',loss=loss_fun(lambd = 0.1),metrics=['accuracy'])
conv_model.summary()
history = conv_model.fit(train_dataset, epochs=40, validation_data=dev_dataset)

def conv_general_drop(input_shape, r):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 4,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Dropout(rate = r)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 8,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.Dropout(rate = r)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 16,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.Dropout(rate = r)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model4 = conv_general_drop((1071,16,1),0.01)
model4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model4.summary()
history = model4.fit(train_dataset, epochs=60, validation_data=dev_dataset)
# best!! 100% train accuracy, 70% dev accuracy
# try adjusting kernel strides and sizes in first conv layer!!

# now let's plot the confusion matrices on the dev set
predictions1 = model1.predict(X_dev)
predictions2 = model2.predict(X_dev)
predictions3 = model3.predict(X_dev)
predictions4 = model4.predict(X_dev)

conf1 = tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(predictions1, axis = 1)).numpy() # rows are real labels, columns are predicted labels
conf2 = tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(predictions2, axis = 1)).numpy()
conf3 = tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(predictions3, axis = 1)).numpy()
conf4 = tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(predictions4, axis = 1)).numpy()
conf1 = conf1/conf1.sum(axis = 1, keepdims = True)
conf2 = conf2/conf2.sum(axis = 1, keepdims = True)
conf3 = conf3/conf3.sum(axis = 1, keepdims = True)
conf4 = conf4/conf4.sum(axis = 1, keepdims = True)
conf_mean = (conf1+conf2+conf3+conf4)/4.0

# create mean model
def mm(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X1 = model1(input_spec)
   X2 = model2(input_spec)
   X3 = model3(input_spec)
   X4 = model4(input_spec)
   output = tf.add_n([X1, X2, X3, X4])/4.0
   return tf.keras.Model(inputs = input_spec,outputs = output)
mean_model = mm((1071, 16, 1))
mean_preds = mean_model(X_dev)
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(mean_preds, axis = 1)).numpy() # rows are real labels, columns are predicted labels

# create max model
def max_m(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X1 = model1(input_spec)
   X2 = model2(input_spec)
   X3 = model3(input_spec)
   X4 = model4(input_spec)
   N = input_spec.shape[0]
   max1 = tf.reshape(tf.reduce_max(X1, axis = 1), shape = (N, 1))
   max2 = tf.reshape(tf.reduce_max(X2, axis = 1), shape = (N, 1))
   max3 = tf.reshape(tf.reduce_max(X3, axis = 1), shape = (N, 1))
   max4 = tf.reshape(tf.reduce_max(X4, axis = 1), shape = (N, 1))
   merged = tf.concat([max1, max2, max3, max4], axis = 1)
   argmax_merged = tf.math.argmax(merged, axis = 1)
   def get_pred(ind, val):
      if ind == 0:
         return X1[val,:]
      if ind == 1:
         return X2[val,:]
      if ind == 2:
         return X3[val,:]
      return X4[val,:]
   concat = tf.concat([get_pred(argmax_merged[i].numpy(),i) for i in range(len(argmax_merged))], axis = 0)
   output = tf.reshape(concat,shape = (N, 4))
   return tf.keras.Model(inputs = input_spec,outputs = output)

def max_m(X):
   X1 = model1(X)
   X2 = model2(X)
   X3 = model3(X)
   X4 = model4(X)
   N = X.shape[0]
   max1 = tf.reshape(tf.reduce_max(X1, axis = 1), shape = (N, 1))
   max2 = tf.reshape(tf.reduce_max(X2, axis = 1), shape = (N, 1))
   max3 = tf.reshape(tf.reduce_max(X3, axis = 1), shape = (N, 1))
   max4 = tf.reshape(tf.reduce_max(X4, axis = 1), shape = (N, 1))
   merged = tf.concat([max1, max2, max3, max4], axis = 1)
   argmax_merged = tf.math.argmax(merged, axis = 1)
   def get_pred(ind, val):
      if ind == 0:
         return X1[val,:]
      if ind == 1:
         return X2[val,:]
      if ind == 2:
         return X3[val,:]
      return X4[val,:]
   concat = tf.concat([get_pred(argmax_merged[i].numpy(),i) for i in range(len(argmax_merged))], axis = 0)
   output = tf.reshape(concat,shape = (N, 4))
   return output

max_preds = max_m(X_dev)
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(max_preds, axis = 1)).numpy() # rows are real labels, columns are predicted labels

def conv_general(input_shape, r, num_first_filters):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = num_first_filters,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   if r != None:
      X = tfl.Dropout(rate = r)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 2*num_first_filters,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   if r != None:
      X = tfl.Dropout(rate = r)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 4*num_first_filters,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   if r != None:
      X = tfl.Dropout(rate = r)(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_general((1071,16,1), None, 2)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# works on training set, not on dev set

model = conv_general((1071,16,1), 0.2, 2)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# not good on either

model = conv_general((1071,16,1), None, 1)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# not good on either

def conv_pipe(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 4,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 4,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 4,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_pipe((1071,16,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=20, validation_data=dev_dataset)

def conv_small(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 4,kernel_size = (322, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,2),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Dropout(rate = 0.3)(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_small((1071,16,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# not bad, but plateau's at about 0.5 for dev set accuracy

def conv_bn_drop_more(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 8,kernel_size = (643, 2),padding = 'same',strides = (10, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (108,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 16,kernel_size = (10, 2),padding = 'same',strides = (10, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (10,2),strides = (1,1),padding = 'same')(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Conv2D(filters = 32,kernel_size = (11, 2),strides = (1, 1))(X)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (1,2),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   X = tfl.Dropout(rate = 0.3)(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_bn_drop_more((1071,16,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# best!! 100% train accuracy, 70% dev accuracy

def conv_big_stride(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 4,kernel_size = (322, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,2),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_big_stride((1071,16,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# good on train, not on dev still (50% ish)

def conv_minimal(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 3,kernel_size = (643, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,16),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal((1071,16,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# bad

# try to decrease data size even more...
X_train = X_train[:,:,range(6,16)]
X_dev = X_dev[:,:,range(6,16)]
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(50)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(50)
def conv_minimal(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 3,kernel_size = (643, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   #X = tfl.MaxPool2D(pool_size = (11,16),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal((1071,10,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# good on train, 60% on dev

def conv_minimal2(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 4,kernel_size = (643, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal2((1071,10,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# same as above

def conv_minimal3(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 10,kernel_size = (150, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,10),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal3((1071,10,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# bad

def conv_minimal4(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 10,kernel_size = (150, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,1),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal4((1071,10,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# good on train, 60% on dev

def conv_minimal5(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 10,kernel_size = (150, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,1),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   X = tfl.Dropout(rate = 0.5)(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal5((1071,10,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# good on train, 60% on dev

def conv_minimal6(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 10,kernel_size = (150, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,1),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax',kernel_regularizer = tf.keras.regularizers.L2(l2 = 10))(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal6((1071,10,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=60, validation_data=dev_dataset)
# good on train, 60% on dev

# get larger data
train_dataset, dev_dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test = generate_data()
model = conv_minimal4((1071,129,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)

def conv_minimal7(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 10,kernel_size = (150, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,1),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   X = tfl.Dropout(rate = 0.5)(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal5((1071,129,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)

X_train = X_train[:,:,range(6,16)]
X_dev = X_dev[:,:,range(6,16)]
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(50)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(50)
def conv_minimal8(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 10,kernel_size = (150, 2),padding = 'same',strides = (100, 1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,1),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax',kernel_regularizer = tf.keras.regularizers.L1(l1 = 0.01))(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal8((1071,10,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)
# good on train, 60% on dev

def conv_minimal_reg(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 10,kernel_size = (150, 2),padding = 'same',strides = (100, 1),kernel_regularizer = tf.keras.regularizers.L2(l2 = 0.1))(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (11,1),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   output = tfl.Dense(4,activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
model = conv_minimal_reg((1071,10,1))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=40, validation_data=dev_dataset)

# current best model
def current_best(rate):
   def mod(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, dropout = rate))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model
   return mod
model = current_best(0.1)((1071,10))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=20, validation_data=dev_dataset)

# try getting a bunch of 100% training set models and take bagged model