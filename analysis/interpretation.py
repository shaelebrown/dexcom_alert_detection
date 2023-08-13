
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

# but this takes a long time per prediction.. about 20 mins for me
prediction = model(X_train[0,:,:].reshape(1,1071,129))
exp = explainer.explain_instance(
    data_row = X_train[0,:,:].reshape(1,1071,129),
    classifier_fn = model.predict)
exp.as_pyplot_figure()
exp.as_map()
plt.show() # doesn't show y labs clearly
# plt.savefig('analysis/sample_explanation.png', bbox_inches="tight") # this does!

t = 0.0026666666666666666 + 0.004666666666666666*np.arange(1071) # consistent across files
f = 187.5*np.arange(129) # consistent across all files

plt.clf()
plt.pcolormesh(t, f, X_dev[0,:,:].T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Actual spectrogram for X_dev[0]')
plt.show()

# now plot with specific coefficients from explainer
exp_spec = np.zeros((1071, 129))
exp_list = exp.as_list()
for (i,j) in exp_list:
    if len(i.split(sep = '<')) <= 2 or len(i.split(sep = '>')) <= 2:
        continue # fix this!
    if not(re.search('0.00 <',i)) or not(re.search('<= 0.00',i)):
        a = i.split(sep = '_')[0].split(sep = ' < ')[1]
        b = i.split(sep = '-')[1].split(sep = ' <= ')[0]
        exp_spec[int(b), int(a)] = j
plt.pcolormesh(t, f, exp_spec.T)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Actual spectrogram for X_dev[0]')
plt.show()


