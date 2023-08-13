
# script for interpretation of bidirectional LSTM model with 512 units, followed
# by two dense layers (of sizes 128 and 4 respectively)

# import modules
import lime
from lime import lime_tabular
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.framework import ops
import keras
import matplotlib.pyplot as plt
import re
import pydub
from pydub import AudioSegment
from pydub.playback import play
import scipy.io.wavfile
from scipy.io.wavfile import read
from scipy import signal
import os

# set random state for reproducibility in python, numpy and tf
tf.keras.utils.set_random_seed(123)

# read in data and model
model = keras.models.load_model('analysis/model.keras')
X_train = np.loadtxt('data/modelling/train.txt').reshape((1600, 1071, 129))
X_dev = np.loadtxt('data/modelling/dev.txt').reshape((200, 1071, 129))
X_test = np.loadtxt('data/modelling/test.txt').reshape((200, 1071, 129))
negative = np.array([1,0,0,0]).reshape((1,4))
high = np.array([0,1,0,0]).reshape((1,4))
low = np.array([0,0,1,0]).reshape((1,4))
urgent_low = np.array([0,0,0,1]).reshape((1,4))
Y_train = np.concatenate([negative for x in range(400)] + [high for x in range(400)] + [low for x in range(400)] + [urgent_low for x in range(400)],axis = 0)
Y_dev = np.concatenate([negative for x in range(50)] + [high for x in range(50)] + [low for x in range(50)] + [urgent_low for x in range(50)],axis = 0)
Y_test = np.concatenate([negative for x in range(50)] + [high for x in range(50)] + [low for x in range(50)] + [urgent_low for x in range(50)],axis = 0)

# shap explainers didn't work for this model.. (LSTM issue)

# possible with LIME, thinking of data as tabular:
class_names=['negative','high','low','urgent_low']
train_labels = np.take_along_axis(arr = np.array(class_names), indices = np.array(np.argmax(Y_train, axis = 1)),axis=0)
explainer = lime_tabular.RecurrentTabularExplainer(
    X_train, training_labels = train_labels,
    mode = "classification",
    feature_names = [str(i) for i in range(1071)],
    discretize_continuous = True,
    class_names = class_names,
    discretizer = 'decile')

# let's look at dev set instance 190 (index 189):
prediction = model.predict(X_dev[189,:,:].reshape((1, 1071, 129)))
print(np.array(class_names)[np.argmax(Y_dev[189,:])]) # real urgent low
print(np.array(class_names)[np.argmax(prediction)]) # predicted negative by model! This is really bad..

# let's listen to the instance:
unclear_alerts = AudioSegment.from_file('data/alerts/under_blanket_alerts.m4a')
unclear_urgent_low = unclear_alerts[11500:14000]
instance = AudioSegment.from_file('data/ESC-50-master/audio/5-204604-A-24.wav')
instance = instance.set_frame_rate(48000)
instance = instance.overlay(unclear_urgent_low, position = 1408)
play(instance)

# verify that the instance is the same as what the model saw:
fname = "data/ESC-50-master/audio/temp.wav"
instance.export(fname,format = "wav")
sr_value, x_value = scipy.io.wavfile.read(fname)
_, _, Sxx= signal.spectrogram(x_value,sr_value)
Sxx = Sxx.swapaxes(0,1)
Sxx = np.expand_dims(Sxx, axis = 0)
os.remove(fname)
np.array_equal(model.predict(Sxx),prediction) # yup exactly the same prediction values
del Sxx, fname, instance, sr_value, x_value, unclear_alerts, unclear_urgent_low

# let's see why the model thought the prediction was negative
# this took about 27 mins for me...
exp = explainer.explain_instance(
    data_row = X_dev[189,:,:].reshape(1,1071,129),
    classifier_fn = model.predict)
exp.as_pyplot_figure()
exp.as_map()
plt.show() # doesn't show y labs clearly
# plt.savefig('analysis/sample_explanation.png', bbox_inches="tight") # this does!

# plot one spectrogram
t = 0.0026666666666666666 + 0.004666666666666666*np.arange(1071) # consistent across files
f = 187.5*np.arange(129) # consistent across all files
plt.clf()
plt.pcolormesh(t, f, X_dev[189,:,:].T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Actual spectrogram for X_dev[189,:,:]')
plt.show()

# now plot spectrogram with specific coefficients from exp
exp_spec = np.zeros((1071, 129))
exp_list = exp.as_list()
for (i,j) in exp_list:
    s = i.split(sep = '_t-')
    a = int(s[0].split(sep = ' ')[-1])
    b = int(s[1].split(sep = ' ')[0])
    exp_spec[b, a] = j  
plt.clf()
plt.pcolormesh(t, f, exp_spec.T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Explanation for X_dev[189,:,:]')
plt.show()