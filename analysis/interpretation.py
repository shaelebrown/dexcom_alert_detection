
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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import math
from scipy.stats import false_discovery_control
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import t
from sklearn.decomposition import PCA
import igraph as ig

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

# what do the training predictions look like?
# clearly they are at most 3D (p0 = 1 - (p1 + p2 + p3))
training_predictions = model.predict(X_train)
p_high = training_predictions[:,1]
p_low = training_predictions[:,2]
p_urgent_low = training_predictions[:,3]

# plot prediction probability PDP's (partial dependence plots)
fig, axs = plt.subplots(3,figsize=(7/3, 7))
axs[0].scatter(x = p_high,y = p_low)
axs[0].set_xlabel('Prob High')
axs[0].set_ylabel('Prob Low')
axs[1].scatter(x = p_high,y = p_urgent_low)
axs[1].set_xlabel('Prob High')
axs[1].set_ylabel('Prob Urgent Low')
axs[2].scatter(x = p_low,y = p_urgent_low)
axs[2].set_xlabel('Prob Low')
axs[2].set_ylabel('Prob Urgent Low')
fig.suptitle('Prediction probability PDPs', fontsize='xx-large')
fig.tight_layout(pad=1.0)
plt.show()

# this shows that (roughly) either Prob High is close to 0, or
# Prob Low = 1 - Prob High (upper plot) and 
# Prob Urgent Low is close to 0

# therefore we can parameterize the whole prediction space by the
# following variables
training_prediction_2D = np.vstack(((1 - p_high)*p_low + p_high*(1 - p_high),(1 - p_high)*p_urgent_low + p_high*(1 - p_urgent_low))).T
plt.clf()
plt.scatter(x = training_prediction_2D[:,0],y = training_prediction_2D[:,1])
plt.ylabel('Embedding dim 2')
plt.xlabel('Embedding dim 1')
plt.title('2D Embedding of model predictions')
plt.show()

# we will split the data into four overlapping regions - triangles of side
# width 0.3 around each vertex of the main train, and a middle region which
# expands until the 0.15 triangles around each vertex:
plt.scatter(x = training_prediction_2D[:,0],y = training_prediction_2D[:,1])
plt.ylabel('Embedding dim 2')
plt.xlabel('Embedding dim 1')
plt.title('Cover of 2D model predictions')
plt.fill_between(np.arange(0.0, 0.3, 0.01), 0.7, 1 - np.arange(0.0, 0.3, 0.01),facecolor='green', interpolate=True, alpha = 0.3)
plt.fill_between(np.arange(0.0, 0.3, 0.01), 0, 0.3 - np.arange(0.0, 0.3, 0.01),facecolor='red', interpolate=True, alpha = 0.3)
plt.fill_between(np.arange(0.7, 1, 0.01), 0, 1 - np.arange(0.7, 1, 0.01),facecolor='blue', interpolate=True, alpha = 0.3)
plt.fill_between(np.arange(0.15, 0.85, 0.01), 0, 1 - np.arange(0.15, 0.85, 0.01),facecolor='gray', interpolate=True, alpha = 0.3)
plt.fill_between(np.arange(0, 0.16, 0.01), 0.15 - np.arange(0, 0.16, 0.01), 0.85,facecolor='gray', interpolate=True, alpha = 0.3)
plt.show()

# now let's get the cover sets
top_cover = np.array(np.where(training_prediction_2D[:,1] >= 0.7)[0])
right_cover = np.array(np.where(training_prediction_2D[:,0] >= 0.7)[0])
left_cover = np.array(np.where(0.3 - training_prediction_2D[:,0] >= 0)[0])
int1 = np.intersect1d(np.array(np.where(training_prediction_2D[:,0] <= 0.85)[0]),np.array(np.where(training_prediction_2D[:,0] >= 0.3)[0]))
int2 = np.intersect1d(np.array(np.where(training_prediction_2D[:,1] <= 0.85)[0]),np.array(np.where(0.3 - training_prediction_2D[:,0] <= training_prediction_2D[:,0])[0]))
middle_cover = np.union1d(int1,int2)
del int1, int2

# now we will extract the final layer activations for mapper
intermediate_output = tf.keras.Model(model.input,model.get_layer('dense').output)
activations = np.array(intermediate_output(X_train))

# subset for preimage sets
top_preimage = activations[top_cover,:]
left_preimage = activations[left_cover,:]
right_preimage = activations[right_cover,:]
middle_preimage = activations[middle_cover,:]

# now we will compute clusterings between 2 and 20 clusters and plot to find
# the elbow points
kmeans_top = [KMeans(n_clusters = nclust, random_state = 123).fit(top_preimage).inertia_ for nclust in range(2,21)]
kmeans_left = [KMeans(n_clusters = nclust, random_state = 123).fit(left_preimage).inertia_ for nclust in range(2,21)]
kmeans_right = [KMeans(n_clusters = nclust, random_state = 123).fit(right_preimage).inertia_ for nclust in range(2,21)]
kmeans_middle = [KMeans(n_clusters = nclust, random_state = 123).fit(middle_preimage).inertia_ for nclust in range(2,21)]
fig, axs = plt.subplots(2, 2)
axs[0,0].scatter(x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],y = kmeans_top)
axs[0,0].set_xlabel('Clusters')
axs[0,0].set_ylabel('WSS')
axs[0,0].set_title('Top Preimage Set')
axs[0,1].scatter(x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],y = kmeans_left)
axs[0,1].set_xlabel('Clusters')
axs[0,1].set_ylabel('WSS')
axs[0,1].set_title('Left Preimage Set')
axs[1,0].scatter(x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],y = kmeans_right)
axs[1,0].set_xlabel('Clusters')
axs[1,0].set_ylabel('WSS')
axs[1,0].set_title('Right Preimage Set')
axs[1,1].scatter(x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],y = kmeans_middle)
axs[1,1].set_xlabel('Clusters')
axs[1,1].set_ylabel('WSS')
axs[1,1].set_title('Middle Preimage Set')
fig.tight_layout(pad=1.0)
plt.show()

# 5 clusters seems sufficient across all preimage sets (and
# based on the small sample size of the dev set), so we'll use that
# in mapper for the dev set analysis
activations_dev = np.array(intermediate_output(X_dev))
predictions_dev = model.predict(X_dev)
p_high = predictions_dev[:,1]
p_low = predictions_dev[:,2]
p_urgent_low = predictions_dev[:,3]
dev_predictions_2D = np.vstack(((1 - p_high)*p_low + p_high*(1 - p_high),(1 - p_high)*p_urgent_low + p_high*(1 - p_urgent_low))).T
top_cover = np.array(np.where(dev_predictions_2D[:,1] >= 0.7)[0])
right_cover = np.array(np.where(dev_predictions_2D[:,0] >= 0.7)[0])
left_cover = np.array(np.where(0.3 - dev_predictions_2D[:,0] >= 0)[0])
int1 = np.intersect1d(np.array(np.where(dev_predictions_2D[:,0] <= 0.85)[0]),np.array(np.where(dev_predictions_2D[:,0] >= 0.3)[0]))
int2 = np.intersect1d(np.array(np.where(dev_predictions_2D[:,1] <= 0.85)[0]),np.array(np.where(0.3 - dev_predictions_2D[:,0] <= dev_predictions_2D[:,0])[0]))
middle_cover = np.union1d(int1,int2)
del int1, int2
top_preimage = activations_dev[top_cover,:]
left_preimage = activations_dev[left_cover,:]
right_preimage = activations_dev[right_cover,:]
middle_preimage = activations_dev[middle_cover,:]
clustering_top = KMeans(n_clusters = 10, random_state = 123).fit(top_preimage)
clustering_left = KMeans(n_clusters = 10, random_state = 123).fit(left_preimage)
clustering_right = KMeans(n_clusters = 10, random_state = 123).fit(right_preimage)
clustering_middle = KMeans(n_clusters = 10, random_state = 123).fit(middle_preimage)
clusters_top = clustering_top.labels_
clusters_left = clustering_left.labels_
clusters_right = clustering_right.labels_
clusters_middle = clustering_middle.labels_
adj = np.zeros((30,30))
overlaps_top_left = np.intersect1d(top_cover,left_cover)
overlaps_top_right = np.intersect1d(top_cover,right_cover)
overlaps_left_right = np.intersect1d(left_cover,right_cover)
overlaps_middle_left = np.intersect1d(middle_cover,left_cover)
overlaps_middle_right = np.intersect1d(middle_cover,right_cover)
overlaps_middle_top = np.intersect1d(middle_cover,top_cover)
edges = np.zeros((40,40))
for o in overlaps_top_left:
    edges[clusters_top[np.where(top_cover == o)[0][0]],10 + clusters_left[np.where(left_cover == o)[0][0]]] = 1
    edges[10 + clusters_left[np.where(left_cover == o)[0][0]],clusters_top[np.where(top_cover == o)[0][0]]] = 1
for o in overlaps_top_right:
    edges[clusters_top[np.where(top_cover == o)[0][0]],20 + clusters_right[np.where(right_cover == o)[0][0]]] = 1
    edges[20 + clusters_right[np.where(right_cover == o)[0][0]],clusters_top[np.where(top_cover == o)[0][0]]] = 1
for o in overlaps_left_right:
    edges[10 + clusters_left[np.where(left_cover == o)[0][0]],20 + clusters_right[np.where(right_cover == o)[0][0]]] = 1
    edges[20 + clusters_right[np.where(right_cover == o)[0][0]],10 + clusters_left[np.where(left_cover == o)[0][0]]] = 1
for o in overlaps_middle_right:
    edges[30 + clusters_middle[np.where(middle_cover == o)[0][0]],20 + clusters_right[np.where(right_cover == o)[0][0]]] = 1
    edges[20 + clusters_right[np.where(right_cover == o)[0][0]],30 + clusters_middle[np.where(middle_cover == o)[0][0]]] = 1
for o in overlaps_middle_left:
    edges[30 + clusters_middle[np.where(middle_cover == o)[0][0]],10 + clusters_left[np.where(left_cover == o)[0][0]]] = 1
    edges[10 + clusters_left[np.where(left_cover == o)[0][0]],30 + clusters_middle[np.where(middle_cover == o)[0][0]]] = 1
for o in overlaps_middle_top:
    edges[30 + clusters_middle[np.where(middle_cover == o)[0][0]],clusters_top[np.where(top_cover == o)[0][0]]] = 1
    edges[clusters_top[np.where(top_cover == o)[0][0]],30 + clusters_middle[np.where(middle_cover == o)[0][0]]] = 1

# create and visualize mapper graph
g = ig.Graph.Adjacency(edges, mode = 'undirected')
vertex_size = [len(np.where(clusters_top == i)[0]) for i in range(10)] + [len(np.where(clusters_left == i)[0]) for i in range(10)] + [len(np.where(clusters_right == i)[0]) for i in range(10)] + [len(np.where(clusters_middle == i)[0]) for i in range(10)]
vertex_size = [i/max(vertex_size) for i in vertex_size]
# color mapping
def rgb_to_hex(r, g, b):
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)
# p_urgent_low is red, p_low is green, p_high is blue
vertex_color = [rgb_to_hex(int(255*p_urgent_low[i]), int(255*p_low[i]), int(255*p_high[i])) for i in range(len(p_low))]
plt.clf()
fig, ax = plt.subplots()
ig.plot(g, target=ax, vertex_size = vertex_size, vertex_color = vertex_color)
plt.show()