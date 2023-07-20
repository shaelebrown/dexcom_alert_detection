
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

# set random state for reproducibility
random.seed(123)

# read in dexcom alert recordings and segment
clear_alerts = AudioSegment.from_file('data/alerts/clear_alerts.m4a')
unclear_alerts = AudioSegment.from_file('data/alerts/under_blanket_alerts.m4a')
clear_high = clear_alerts[1000:4000]
clear_low = clear_alerts[7650:9650]
clear_urgent_low = clear_alerts[14500:17000]
unclear_high = unclear_alerts[700:3700]
unclear_low = unclear_alerts[6000:8000]
unclear_urgent_low = unclear_alerts[11500:14000]

# splitting data into train, test and dev sets
if '.DS_Store' in os.listdir('/Users/jibaccount/Downloads/ESC-50-master/audio'):
    os.remove('/Users/jibaccount/Downloads/ESC-50-master/audio/.DS_Store')
if 'temp.wav' in os.listdir('/Users/jibaccount/Downloads/ESC-50-master/audio'):
    os.remove('/Users/jibaccount/Downloads/ESC-50-master/audio/temp.wav')
fnames = os.listdir('/Users/jibaccount/Downloads/ESC-50-master/audio')
esc_meta = pd.read_csv("/Users/jibaccount/Downloads/ESC-50-master/meta/esc50.csv")
train, subset = train_test_split(fnames, test_size = 0.2, random_state = 123)
dev, test = train_test_split(subset, test_size = 0.5, random_state = 123)
del subset

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

train_negative, train_high, train_low, train_urgent_low, train_unclear_high, train_unclear_low, train_unclear_urgent_low = split_into_eights(train)
dev_negative, dev_high, dev_low, dev_urgent_low, dev_unclear_high, dev_unclear_low, dev_unclear_urgent_low = split_into_eights(dev)
test_negative, test_high, test_low, test_urgent_low, test_unclear_high, test_unclear_low, test_unclear_urgent_low = split_into_eights(test)

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
train_X = np.concatenate([train_negative, train_high, train_unclear_high, train_low, train_unclear_low, train_urgent_low, train_unclear_urgent_low], axis = 0)
dev_X = np.concatenate([dev_negative, dev_high, dev_unclear_high, dev_low, dev_unclear_low, dev_urgent_low, dev_unclear_urgent_low], axis = 0)
test_X = np.concatenate([test_negative, test_high, test_unclear_high, test_low, test_unclear_low, test_urgent_low, test_unclear_urgent_low], axis = 0)

# labels are one-hot encoded vectors from 4 classes
negative = np.array([1,0,0,0]).reshape((4,1))
high = np.array([0,1,0,0]).reshape((4,1))
low = np.array([0,0,1,0]).reshape((4,1))
urgent_low = np.array([0,0,0,1]).reshape((4,1))
train_Y = np.concatenate([negative for x in range(400)] + [high for x in range(400)] + [low for x in range(400)] + [urgent_low for x in range(400)],axis = 1)
dev_Y = np.concatenate([negative for x in range(50)] + [high for x in range(50)] + [low for x in range(50)] + [urgent_low for x in range(50)],axis = 1)
test_Y = np.concatenate([negative for x in range(50)] + [high for x in range(50)] + [low for x in range(50)] + [urgent_low for x in range(50)],axis = 1)
