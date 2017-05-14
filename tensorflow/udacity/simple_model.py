from __future__ import print_function
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import sys
import tarfile
from PIL import Image
from scipy import ndimage
from scipy import stats
from sklearn import linear_model
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sets import Set
from sklearn.metrics import accuracy_score

compressed_pickle_file = 'notMNIST_unique.pickle'
data_root = '.'

def read_into_memory(file_path):
  if not os.path.exists(file_path):
    print ('Could not find pickle file at %s' % file_path)
  else:
    try:
      f = open(file_path, 'rb')
      dataset = pickle.load(f)
      return dataset
    except Exception as e:
      print ('Could not read file %s due to %s' % (file_path, e))

dataset = read_into_memory(os.path.join(data_root, compressed_pickle_file))

train_data = dataset['train_dataset']
train_labels = dataset['train_labels']
valid_data = dataset['valid_dataset']
valid_labels = dataset['valid_labels']

# Sanity checking
print (train_data.shape)
print (train_labels.shape)

# Reduce the size on which we train, in the interest of time.
train_size = 50000
validation_size = 5000

def resize_data(data, n):
  return data[:n]

resized_train_data = resize_data(train_data, train_size)
resized_valid_data = resize_data(valid_data, validation_size)
resized_train_labels = resize_data(train_labels, train_size)
resized_valid_labels = resize_data(valid_labels, validation_size)
print (resized_train_data.shape)
print (resized_valid_data.shape)

# Reshape training data.
def reshape_data(data):
  shape_x = data.shape[0]
  shape_y = data.shape[1] ** 2
  new_shape = (shape_x, shape_y)
  reshaped_data = np.ndarray(shape=(new_shape))
  for i, image in enumerate(data):
    reshaped_data[i] = np.reshape(data[i], (shape_y, ))
  return reshaped_data

reshaped_train_data = reshape_data(resized_train_data)
reshaped_valid_data = reshape_data(resized_valid_data)
print (reshaped_train_data.shape)
print (reshaped_valid_data.shape)

def get_svm_predictions():
  model = svm.SVC(gamma=0.001, C=100.)
  model.fit(reshaped_train_data, resized_train_labels)
  predictions = model.predict(reshaped_valid_data)
  print (predictions)
  return predictions

def get_lr_predictions():
  model = linear_model.LogisticRegression(C=1e5)
  model.fit(reshaped_train_data, resized_train_labels)
  predictions = model.predict(reshaped_valid_data)
  print (predictions)
  return predictions

svm_predictions = get_svm_predictions()
lr_predictions = get_lr_predictions()

# Time to check correctness
print (accuracy_score(resized_valid_labels, svm_predictions))
print (accuracy_score(resized_valid_labels, lr_predictions))

def show_errors(labels, predictions, max_images):
  for i, (label, prediction) in enumerate(zip(labels, predictions)):
    if label != prediction and i < max_images:
      plt.imshow(resized_valid_data[i])
      plt.title('Actual: %d, Predicted: %s' % (labels[i], predictions[i]))
      plt.show()

show_errors(resized_valid_labels, lr_predictions, 2)
