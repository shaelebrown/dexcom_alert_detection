
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
import matplotlib as mpl
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
import time
from scipy.spatial import distance
import math
from mpl_toolkits.mplot3d import Axes3D
import kmapper as km

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

plt.pcolormesh(range(16), range(1071), X_train[0,:,range(16)].T, shading='gouraud') #0-399 for negative, 400-799 for high, 800-1199 for low and 1200-1599 for urgent low
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Negative example')
plt.show()

plt.pcolormesh(range(16), range(1071), X_train[500,:,range(16)].T, shading='gouraud') #0-399 for negative, 400-799 for high, 800-1199 for low and 1200-1599 for urgent low
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('High example')
plt.show()

plt.pcolormesh(range(16), range(1071), X_train[800,:,range(16)].T, shading='gouraud') #0-399 for negative, 400-799 for high, 800-1199 for low and 1200-1599 for urgent low
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Low example')
plt.show()

plt.pcolormesh(range(16), range(1071), X_train[1200,:,range(16)].T, shading='gouraud') #0-399 for negative, 400-799 for high, 800-1199 for low and 1200-1599 for urgent low
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Urgent low example')
plt.show()

# length for each alert:
# high: 3s
# low: 2s
# urgent low: 2.5s
# clip length was 5s, so each time step was 5/1071 = 0.00467s
# so each alert in time steps is:
# high: 643
# low: 429
# urgent low: 536

# frequency range for each alert:
# high: 13-15
# low: 13-15
# urgent_low: 7-10

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

# now let's plot the confusion matrices
predictions_dev = model.predict(X_dev)
predictions_train = model.predict(X_train)
conf_dev = tf.math.confusion_matrix(labels = np.argmax(Y_dev, axis = 1),predictions = np.argmax(predictions_dev, axis = 1)).numpy() # rows are real labels, columns are predicted labels
conf_train = tf.math.confusion_matrix(labels = np.argmax(Y_train, axis = 1),predictions = np.argmax(predictions_train, axis = 1)).numpy()
conf_dev = conf_dev/conf_dev.sum(axis = 1, keepdims = True)
conf_train = conf_train/conf_train.sum(axis = 1, keepdims = True)
plt.clf()
fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.flip(conf_train, axis = 0), cmap='hot', interpolation='nearest')
axs[0].set_title('Training set confusion matrix')
axs[1].imshow(np.flip(conf_dev, axis = 0), cmap='hot', interpolation='nearest')
axs[1].set_title('Dev set confusion matrix')
axs[0].set_xlabel('Predicted label')
axs[1].set_xlabel('Predicted label')
axs[0].set_ylabel('Actual label')
axs[1].set_ylabel('Actual label')
axs[0].set_xticks(np.array([0,1,2,3]),np.array(['None', 'High', 'Low', 'Urgent low']))
axs[1].set_xticks(np.array([0,1,2,3]),np.array(['None', 'High', 'Low', 'Urgent low']))
axs[0].set_yticks(np.array([3,2,1,0]),np.array(['None', 'High', 'Low', 'Urgent low']))
axs[1].set_yticks(np.array([3,2,1,0]),np.array(['None', 'High', 'Low', 'Urgent low']))
fig.tight_layout(pad=1.0)
plt.show()
# on dev set the model is less good at correctly predicting lows and negatives
# also the model predicts low for more of both negatives and highs, so what are some examples?
incorrect_pred_low = np.intersect1d(np.where(np.argmax(Y_dev, axis = 1) != 2), np.where(np.argmax(predictions_dev, axis = 1) == 2))
for ind in incorrect_pred_low:
    audio = AudioSegment.from_wav('data/modelling/modified_audio_files/dev_' + str(ind) + '.wav')
    play(audio)
    time.sleep(1)
# mainly no alert or quiet (muffled) high alert

# what do the training predictions look like?
# clearly they are at most 3D (p0 = 1 - (p1 + p2 + p3))
p_high = predictions_train[:,1]
p_low = predictions_train[:,2]
p_urgent_low = predictions_train[:,3]

# plot in 3D (4D not necessary due to linear dependence)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p_high, p_low, p_urgent_low, color='green')
ax.set_xlabel('Prob High')
ax.set_ylabel('Prob Low')
ax.set_zlabel('Prob Urgent Low')
plt.show()

# plot mapper of predictions
# figure out how to do this with density lens function! in place of projected_data
data = np.vstack([p_high, p_low, p_urgent_low]).T
mapper = km.KeplerMapper(verbose = 1)
projected_data = mapper.fit_transform(data, projection = 'knn_distance_5') # 3-NN distance
cover = km.Cover(n_cubes = 10)
graph = mapper.map(projected_data, data, cover = cover)
graph['links'] # gives graph edges
mapper.visualize(graph, title = "Model predictions")

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
activations = np.array(intermediate_output.predict(X_train))

# subset for preimage sets
top_preimage = activations[top_cover,:]
left_preimage = activations[left_cover,:]
right_preimage = activations[right_cover,:]
middle_preimage = activations[middle_cover,:]

# now we will compute clusterings between 2 and 20 clusters and plot to find
# the elbow points
GMM_top = [GaussianMixture(n_components = nclust, random_state = 123).fit(top_preimage).bic(top_preimage) for nclust in range(2,21)]
GMM_left = [GaussianMixture(n_components = nclust, random_state = 123).fit(left_preimage).bic(left_preimage) for nclust in range(2,21)]
GMM_right = [GaussianMixture(n_components = nclust, random_state = 123).fit(right_preimage).bic(right_preimage) for nclust in range(2,21)]
GMM_middle = [GaussianMixture(n_components = nclust, random_state = 123).fit(middle_preimage).bic(middle_preimage) for nclust in range(2,21)]
fig, axs = plt.subplots(2, 2)
axs[0,0].scatter(x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],y = GMM_top)
axs[0,0].set_xlabel('Clusters')
axs[0,0].set_ylabel('BIC')
axs[0,0].set_title('Top Preimage Set')
axs[0,1].scatter(x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],y = GMM_left)
axs[0,1].set_xlabel('Clusters')
axs[0,1].set_ylabel('BIC')
axs[0,1].set_title('Left Preimage Set')
axs[1,0].scatter(x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],y = GMM_right)
axs[1,0].set_xlabel('Clusters')
axs[1,0].set_ylabel('BIC')
axs[1,0].set_title('Right Preimage Set')
axs[1,1].scatter(x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],y = GMM_middle)
axs[1,1].set_xlabel('Clusters')
axs[1,1].set_ylabel('BIC')
axs[1,1].set_title('Middle Preimage Set')
fig.tight_layout(pad=1.0)
plt.show()

# it looked like the optimal #'s of clusters were
# 9 for top
# 10 for left
# 5 for right, and
# 7 for middle
clusters_top = GaussianMixture(n_components = 9, random_state = 123).fit(top_preimage)
clusters_left = GaussianMixture(n_components = 10, random_state = 123).fit(left_preimage)
clusters_right = GaussianMixture(n_components = 5, random_state = 123).fit(right_preimage)
clusters_middle = GaussianMixture(n_components = 7, random_state = 123).fit(middle_preimage)

# now we will use these clusters
# in dev set mapper analysis
activations_dev = np.array(intermediate_output.predict(X_dev))
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
# predict maximum cluster probability of each data point in each preimage set
cov_inv_top = [scipy.linalg.pinv(clusters_top.covariances_[i]) for i in range(9)]
cov_inv_left = [scipy.linalg.pinv(clusters_left.covariances_[i]) for i in range(10)]
cov_inv_right = [scipy.linalg.pinv(clusters_right.covariances_[i]) for i in range(5)]
cov_inv_middle = [scipy.linalg.pinv(clusters_middle.covariances_[i]) for i in range(7)]
det_top = [scipy.linalg.det(clusters_top.covariances_[i]) for i in range(9)]
det_left = [scipy.linalg.det(clusters_left.covariances_[i]) for i in range(10)]
det_right = [scipy.linalg.det(clusters_right.covariances_[i]) for i in range(5)]
det_middle = [scipy.linalg.det(clusters_middle.covariances_[i]) for i in range(7)]
# use 1/(sqrt(2*pi*det(cov)))*exp(-d^2/2) where d is mahalanobis distance
clusters_top.means_.shape
clusters_top.covariances_.shape
scipy.linalg.pinv(clusters_top.covariances_[0]).shape
check = clusters_top
clustering_top = KMeans(n_clusters = 10, random_state = 123).fit(top_preimage)
clustering_left = KMeans(n_clusters = 10, random_state = 123).fit(left_preimage)
clustering_right = KMeans(n_clusters = 10, random_state = 123).fit(right_preimage)
clustering_middle = KMeans(n_clusters = 10, random_state = 123).fit(middle_preimage)
clusters_top = clustering_top.labels_
clusters_left = clustering_left.labels_
clusters_right = clustering_right.labels_
clusters_middle = clustering_middle.labels_
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
layout = g.layout(layout='auto')
vertex_size = [len(np.where(clusters_top == i)[0]) for i in range(10)] + [len(np.where(clusters_left == i)[0]) for i in range(10)] + [len(np.where(clusters_right == i)[0]) for i in range(10)] + [len(np.where(clusters_middle == i)[0]) for i in range(10)]
vertex_size = [i/max(vertex_size) for i in vertex_size]
# color mapping
def rgb_to_hex(r, g, b):
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)
# p_urgent_low is red, p_low is green, p_high is blue
vertex_color = [rgb_to_hex(int(255*p_urgent_low[i]), int(255*p_low[i]), int(255*p_high[i])) for i in range(len(p_low))]
high_color = [rgb_to_hex(0, 0, int(255*p_high[i])) for i in range(len(p_high))]
low_color = [rgb_to_hex(0, int(255*p_low[i]), 0) for i in range(len(p_low))]
urgent_low_color = [rgb_to_hex(int(255*p_urgent_low[i]), 0, 0) for i in range(len(p_urgent_low))]
p_negative = predictions_dev[:,0]
negative_color = [rgb_to_hex(int(255*p_negative[i]), int(255*p_negative[i]), int(255*p_negative[i])) for i in range(len(p_urgent_low))]
fourD_preds = np.vstack([p_high, p_low, p_urgent_low, p_negative]).T
cols = ['#FF0000', '#00FF00', '#0000FF', '#FFFFFF']
max_pred_color = np.argmax(fourD_preds, axis = 1)
consensus_color = [cols[i] for i in max_pred_color]
labels = ["t" + str(i) for i in range(10)] + ["l" + str(i) for i in range(10)] + ["r" + str(i) for i in range(10)] + ["m" + str(i) for i in range(10)]
plt.clf()
fig, ax = plt.subplots(2, 2)
ax[0,0].set_title('Probability high alert')
ax[0,1].set_title('Probability no alert')
ax[1,0].set_title('Probability low alert')
ax[1,1].set_title('Probability urgent low alert')
ig.plot(g, target=ax[0,0], vertex_size = vertex_size, vertex_color = high_color, layout = layout)
ig.plot(g, target=ax[0,1], vertex_size = vertex_size, vertex_color = negative_color, layout = layout)
ig.plot(g, target=ax[1,0], vertex_size = vertex_size, vertex_color = low_color, layout = layout)
ig.plot(g, target=ax[1,1], vertex_size = vertex_size, vertex_color = urgent_low_color, layout = layout)
plt.show()

# plot single graph colored by majority probability
plt.clf()
fig, ax = plt.subplots()
ax.set_title('Largest class probability')
ig.plot(g, target = ax, vertex_size = vertex_size, vertex_color = consensus_color, layout = layout)
plt.colorbar(ax = ax)
cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'white']).with_extremes(over='0.25', under='0.75'))
bounds = [0,1,2,3,4]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap = cmap, norm = norm),ticks = [0, 0.5, 1.5, 2.5, 3.5])
cbar.ax.set_yticklabels(['', 'High', 'Low', 'Urgent Low', 'Negative']) 
plt.show()

# play audio clips in different vertices
vertex = 0 # first top class (0)
inds = top_cover[np.where(clusters_top == 0)]
for ind in inds:
    audio = AudioSegment.from_wav('data/modelling/modified_audio_files/dev_' + str(ind) + '.wav')
    play(audio)
    time.sleep(1)
# plays in a loop
    
# subset dataset and visualize for custom convolution layers
X_train = X_train[:,:,[7,8,9,10,13,14,15]]
X_dev = X_dev[:,:,[7,8,9,10,13,14,15]]

plt.pcolormesh(range(7), range(1071), X_train[0,:,range(7)].T, shading='gouraud') #0-399 for negative, 400-799 for high, 800-1199 for low and 1200-1599 for urgent low
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Negative example')
plt.show()

plt.pcolormesh(range(7), range(1071), X_train[500,:,range(7)].T, shading='gouraud') #0-399 for negative, 400-799 for high, 800-1199 for low and 1200-1599 for urgent low
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('High example')
plt.show()

plt.pcolormesh(range(7), range(1071), X_train[800,:,range(7)].T, shading='gouraud') #0-399 for negative, 400-799 for high, 800-1199 for low and 1200-1599 for urgent low
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Low example')
plt.show()

plt.pcolormesh(range(7), range(1071), X_train[1200,:,range(7)].T, shading='gouraud') #0-399 for negative, 400-799 for high, 800-1199 for low and 1200-1599 for urgent low
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Urgent low example')
plt.show()

# captures all of the alerts
# 4-6 and 692-894 for high in X_train[500]
# 4-6 and 74-397 for low in X_train[800]
# 0-2 and 368-764 for urgent low in X_train[1200]

# therefore let's do kernels which are initialized as
# X_train[500,593:993,4:6]
# X_train[800,35:435,4:6]
# X_train[1200,366:766,0:2]
