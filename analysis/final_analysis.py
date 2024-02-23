
# script for finaly modelling of dexcom alert data

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
import matplotlib.pyplot as plt

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

# read in sound to go between audio clips
meep = AudioSegment.from_file('data/alerts/meep.m4a')
meep = meep[1000:2000]

# functions

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
    if type == 'dev':
        return Sxx, temp
    return Sxx

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

# function to create training, dev and test datasets spit by 8ths
# also generate audio file for dev set
def generate_data():
    # splitting data into train, test and dev sets
    fnames = np.random.permutation(os.listdir('data/ESC-50-master/audio')).tolist()
    train, subset = train_test_split(fnames, test_size = 0.2, random_state = 123)
    dev, test = train_test_split(subset, test_size = 0.5, random_state = 123)
    del subset
    train_negative, train_high, train_low, train_urgent_low, train_unclear_high, train_unclear_low, train_unclear_urgent_low = split_into_eights(train)
    dev_negative, dev_high, dev_low, dev_urgent_low, dev_unclear_high, dev_unclear_low, dev_unclear_urgent_low = split_into_eights(dev)
    test_negative, test_high, test_low, test_urgent_low, test_unclear_high, test_unclear_low, test_unclear_urgent_low = split_into_eights(test)
    # overlay correct alerts over each audio clip, convert to spectrograms
    # and concatenate
    # also save dev set audio
    train_negative = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = None, type = 'train', index = train_negative.index(f)) for f in train_negative],axis = 0)
    train_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low', type = 'train', index = train_low.index(f) + 800) for f in train_low],axis = 0)
    train_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high', type = 'train', index = train_high.index(f) + 400) for f in train_high],axis = 0)
    train_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low', type = 'train', index = train_urgent_low.index(f) + 1200) for f in train_urgent_low],axis = 0)
    train_unclear_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low', type = 'train', index = train_unclear_low.index(f) + 1000) for f in train_unclear_low],axis = 0)
    train_unclear_high = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high', type = 'train', index = train_unclear_high.index(f) + 600) for f in train_unclear_high],axis = 0)
    train_unclear_urgent_low = np.concatenate([spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low', type = 'train', index = train_unclear_urgent_low.index(f) + 1400) for f in train_unclear_urgent_low],axis = 0)
    dev_negative = [spectrogram('data/ESC-50-master/audio/' + f, alert = None, type = 'dev', index = dev_negative.index(f)) for f in dev_negative]
    audio_dev_negative = [dev_negative[i][1] for i in range(len(dev_negative))]
    dev_negative = np.concatenate([dev_negative[i][0] for i in range(len(dev_negative))],axis = 0)
    dev_low = [spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_low', type = 'dev', index = dev_low.index(f) + 100) for f in dev_low]
    audio_dev_low = [dev_low[i][1] for i in range(len(dev_low))]
    dev_low = np.concatenate([dev_low[i][0] for i in range(len(dev_low))],axis = 0)
    dev_high = [spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_high', type = 'dev', index = dev_high.index(f) + 50) for f in dev_high]
    audio_dev_high = [dev_high[i][1] for i in range(len(dev_high))]
    dev_high = np.concatenate([dev_high[i][0] for i in range(len(dev_high))],axis = 0)
    dev_urgent_low = [spectrogram('data/ESC-50-master/audio/' + f, alert = 'clear_urgent_low', type = 'dev', index = dev_urgent_low.index(f) + 150) for f in dev_urgent_low]
    audio_dev_urgent_low = [dev_urgent_low[i][1] for i in range(len(dev_urgent_low))]
    dev_urgent_low = np.concatenate([dev_urgent_low[i][0] for i in range(len(dev_urgent_low))],axis = 0)
    dev_unclear_low = [spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_low', type = 'dev', index = dev_unclear_low.index(f) + 125) for f in dev_unclear_low]
    audio_dev_unclear_low = [dev_unclear_low[i][1] for i in range(len(dev_unclear_low))]
    dev_unclear_low = np.concatenate([dev_unclear_low[i][0] for i in range(len(dev_unclear_low))],axis = 0)
    dev_unclear_high = [spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_high', type = 'dev', index = dev_unclear_high.index(f) + 75) for f in dev_unclear_high]
    audio_dev_unclear_high = [dev_unclear_high[i][1] for i in range(len(dev_unclear_high))]
    dev_unclear_high = np.concatenate([dev_unclear_high[i][0] for i in range(len(dev_unclear_high))],axis = 0)
    dev_unclear_urgent_low = [spectrogram('data/ESC-50-master/audio/' + f, alert = 'unclear_urgent_low', type = 'dev', index = dev_unclear_urgent_low.index(f) + 175) for f in dev_unclear_urgent_low]
    audio_dev_unclear_urgent_low = [dev_unclear_urgent_low[i][1] for i in range(len(dev_unclear_urgent_low))]
    dev_unclear_urgent_low = np.concatenate([dev_unclear_urgent_low[i][0] for i in range(len(dev_unclear_urgent_low))],axis = 0)
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
    # subset in frequency domain
    X_train = X_train[:,:,[7,8,9,10,13,14,15]]
    X_dev = X_dev[:,:,[7,8,9,10,13,14,15]]
    X_test = X_test[:,:,[7,8,9,10,13,14,15]]
    # labels are one-hot encoded vectors from 4 classes
    negative = np.array([1,0,0,0]).reshape((1,4))
    high = np.array([0,1,0,0]).reshape((1,4))
    low = np.array([0,0,1,0]).reshape((1,4))
    urgent_low = np.array([0,0,0,1]).reshape((1,4))
    Y_train = np.concatenate([negative for x in range(400)] + [high for x in range(400)] + [low for x in range(400)] + [urgent_low for x in range(400)],axis = 0)
    Y_dev = np.concatenate([negative for x in range(50)] + [high for x in range(50)] + [low for x in range(50)] + [urgent_low for x in range(50)],axis = 0)
    Y_test = np.concatenate([negative for x in range(50)] + [high for x in range(50)] + [low for x in range(50)] + [urgent_low for x in range(50)],axis = 0)
    # shuffle dev audio and labels, concatenate audio
    perm = np.random.permutation(200)
    dev_audio_labels = ['negative' for x in range(50)] + ['high' for x in range(50)] + ['low' for x in range(50)] + ['urgent_low' for x in range(50)]
    dev_audio_labels = [dev_audio_labels[i] for i in perm]
    dev_audio_list = audio_dev_negative + audio_dev_high + audio_dev_low + audio_dev_urgent_low + audio_dev_unclear_high + audio_dev_unclear_low + audio_dev_unclear_urgent_low
    dev_audio_list = [dev_audio_list[i] for i in perm]
    dev_audio = pydub.AudioSegment.empty()
    for i in range(200):
        dev_audio = dev_audio.append(dev_audio_list[i], crossfade = 100*int(i > 0))
        dev_audio = dev_audio.append(globals()['meep'])
    # custom convolution initializer from training set
    c1 = X_train[500,413:788,4:6]
    c2 = X_train[800,68:443,4:6]
    c3 = X_train[1200,375:750,0:2]
    init = tf.constant_initializer([c1, c2, c3])
    # permute training set to balance by label
    perm = [[i, 200 + i, 400 + i, 600 + i, 800 + i, 1000 + i, 1200 + i, 1400 + i] for i in range(200)]
    perm = sum(perm, [])
    X_train = tf.gather(X_train, perm)
    Y_train = tf.gather(Y_train, perm)
    train_dataset_big = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    perm = [x for x in range(1600) if x % 8 != 0]
    X_train = tf.gather(X_train, perm)
    Y_train = tf.gather(Y_train, perm)
    train_dataset_small = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(56) # remove one negative example from each batch
    dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(50)
    # return
    return train_dataset_big, train_dataset_small, init, dev_dataset, X_dev, Y_dev, X_test, Y_test, dev_audio, dev_audio_labels

train_dataset_big, train_dataset_small, init, dev_dataset, X_dev, Y_dev, X_test, Y_test, dev_audio, dev_audio_labels = generate_data()

# pretrain cnn and rnn models on small datasets
def cnn(input_shape):
   input_spec = tf.keras.Input(shape = input_shape)
   X = tfl.Conv2D(filters = 3,kernel_size = (375, 2),padding = 'same',strides = (100, 1),kernel_initializer = init)(input_spec)
   X = tfl.ReLU()(X)
   X = tfl.MaxPool2D(pool_size = (4,1),strides = (1,1))(X)
   X = tfl.BatchNormalization()(X)
   X = tfl.Flatten()(X)
   X = tfl.Dense(32, activation = 'tanh')(X)
   X = tfl.Dense(16, activation = 'tanh')(X)
   X = tfl.Dense(8, activation = 'tanh')(X)
   output = tfl.Dense(4, activation = 'softmax')(X)
   return tf.keras.Model(inputs = input_spec,outputs = output)
cnn_model = cnn((1071,7,1))
cnn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn_model.summary()
cnn_model.fit(train_dataset_small, epochs=40, validation_data=dev_dataset)

def rnn(input_shape):
    input_spec = tf.keras.Input(shape = input_shape)
    X = tfl.Bidirectional(tfl.LSTM(units = 512, return_sequences = False, dropout = 0.1))(input_spec)
    X = tfl.Dense(128, activation = 'tanh')(X)
    outputs = tfl.Dense(4, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = input_spec, outputs = outputs)
    return model
rnn_model = rnn((1071,7))
rnn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
rnn_model.summary()
rnn_model.fit(train_dataset_small, epochs=40, validation_data=dev_dataset)

# set both to non-trainable
cnn_model.trainable = False
rnn_model.trainable = False

# mixed fine-tuned model on big dataset
def mixed(input_shape, rnn_model, cnn_model):
   input_spec = tf.keras.Input(shape = input_shape)
   output_rnn = rnn_model(input_spec)
   output_cnn = cnn_model(input_spec)
   joint = tfl.Concatenate(axis = 1)([output_rnn, output_cnn])
   output = tfl.Dense(4, activation = 'softmax')(joint)
   return tf.keras.Model(inputs = input_spec,outputs = output)
mixed_model = mixed((1071,7,1), rnn_model, cnn_model)
mixed_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
mixed_model.summary()
mixed_model.fit(train_dataset_big, epochs=40, validation_data=dev_dataset)
tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(mixed_model(X_dev), axis = 1)) # rows are real labels, columns are predicted labels
