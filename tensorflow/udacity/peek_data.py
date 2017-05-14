from __future__ import print_function
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
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sets import Set
import download_extract_process

# Make sure data is available
download_extract_process.prepare_data()

pickle_file_path = download_extract_process.get_pickle_file_name()
data_root = download_extract_process.get_data_root()

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

dataset = read_into_memory(os.path.join(data_root, pickle_file_path))

# Print all keys to ensure that the pickle was read correctly.
for key, _ in dataset.items():
  print (key)

def plot_images(dataset, n):
  """The shape of the dataset is assumed to be (N, A, B) where N is the number of images"""
  for i in range(0, n):
    index = np.random.randint(0, dataset.shape[0])
    plt.imshow(dataset[index, :, :], cmap="gray")
    plt.show()

# TODO(atri): Again, this should be a flag.
num_images_to_plot = 0

# Plot a few images from all datasets, to ensure that the pixels were read
# correctly.
plot_images(dataset['train_dataset'], num_images_to_plot)
plot_images(dataset['valid_dataset'], num_images_to_plot)
plot_images(dataset['test_dataset'], num_images_to_plot)

def get_hash(image):
  """image is an np array with shape K, K"""
  pil_image = Image.fromarray(image)
  pil_image = np.reshape(pil_image.resize((8, 8)).getdata(), (64, ))
  avg = np.average(pil_image)
  bits = "".join(map(lambda pixel: '1' if pixel < avg else '0', pil_image))
  # Convert the bits into hexadecimal.
  return int(bits, 2).__format__('016x').upper()

def get_unique_images(images):
  """
  Ideally it should use a hashing function that gives the same hash to
  images that look alike
  """
  image_hash_map = {}
  for i in range(0, len(images)):
    image_hash_map[get_hash(images[i])] = i

  return image_hash_map.values()

def get_unique_dataset(dataset):
  save = {}
  unique_images = get_unique_images(dataset['train_dataset'])
  save['train_dataset'] = dataset['train_dataset'][unique_images]
  save['train_labels'] = dataset['train_labels'][unique_images]

  unique_images = get_unique_images(dataset['valid_dataset'])
  save['valid_dataset'] = dataset['valid_dataset'][unique_images]
  save['valid_labels'] = dataset['valid_labels'][unique_images]

  unique_images = get_unique_images(dataset['test_dataset'])
  save['test_dataset'] = dataset['test_dataset'][unique_images]
  save['test_labels'] = dataset['test_labels'][unique_images]
  return save

unique_data = get_unique_dataset(dataset)
for key, data in unique_data.items():
  print (data.shape)

unique_image_pickle_filename = 'notMNIST_unique.pickle'
def save_pickle_file():
  file_path = os.path.join(data_root, unique_image_pickle_filename)
  if os.path.exists(file_path):
    print ("Pickle file %s already exists. Not creating again" % file_path)
  else:
    try:
      save = get_unique_dataset(dataset)
      f = open(file_path, 'wb')
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
      print ('Successfully created file %s' % file_path)
    except Exception as e:
      print ('Could not write file %s due to %s' % (file_path, e))
      raise
  return file_path

compressed_file_path = save_pickle_file()

