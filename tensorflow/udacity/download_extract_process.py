"""Downloads notMNIST dataset, sanitizes it and writes it to a pickle file."""

from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve


def get_data_root():
  return '.'


def get_pickle_file_name():
  return 'notMNIST.pickle'

_URL = 'http://commondatastorage.googleapis.com/books1000/'
_LAST_PERCENTAGE_REPORTED = None


def _download_progress_hook(count, block_size, total_size):
  percent = count * block_size * 100 / total_size
  global _LAST_PERCENTAGE_REPORTED
  if percent != _LAST_PERCENTAGE_REPORTED:
    if percent % 5 == 0:
      sys.stdout.write('%s%%' % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write('.')
      sys.stdout.flush()

  _LAST_PERCENTAGE_REPORTED = percent


def _maybe_download(filename, expected_bytes, force=False):
  dest_filename = os.path.join(get_data_root(), filename)

  if force or not os.path.exists(dest_filename):
    print('Attempting to download file')
    filename, _ = urlretrieve(
        _URL + filename, dest_filename, reporthook=_download_progress_hook)
    print('Download complete')

  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print ('File verified.')
  else:
    print ('File seems to be invalid. Size is %d' % statinfo.st_size)

  return dest_filename


def _maybe_extract(filename, num_classes, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]
  print ('Attempting to extract %s in %s' % (filename, root))
  if os.path.isdir(root) and not force:
    print ('%s already exists. Not extracting %s again' % (root, filename))
  else:
    print ('Extracting %s in %s' % (filename, root))
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(get_data_root())
    tar.close()

  data_folders = [
      os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception('Expected %d folders, found %d instead' % num_classes, len(data_folders))

  print(data_folders)
  return data_folders


_IMAGE_SIZE = 28
_PIXEL_DEPTH = 255.0


def _load_letter(folder, min_num_images):
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
  print ('Processing images from %s' % folder)
  num_images = 0
  for image in image_files:
    image_path = os.path.join(folder, image)
    try:
      image_data = (
          ndimage.imread(image_path).astype(float) - _PIXEL_DEPTH / 2) / _PIXEL_DEPTH
      if image_data.shape != (_IMAGE_SIZE, _IMAGE_SIZE):
        raise Exception('Unexpcted image shape %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print ('Could not read image %s due to %s. Skipped' % (image_path, e))

  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception(
        'Expected to read at least %d images, read only %d' % min_num_images, num_images)

  print ('Full dataset tensor:', dataset.shape)
  print ('Mean:', np.mean(dataset))
  print ('Standard deviation:', np.std(dataset))
  return dataset


def _maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      print ('Found %s, not pickling again' % set_filename)
    else:
      print ('Pickling %s' % set_filename)
      data = _load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print ('Unable to save data to', set_filename, ':', e)
  return dataset_names


def _make_arrays(nb_size, image_size):
  if nb_size:
    dataset = np.ndarray((nb_size, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(nb_size, dtype=np.int32)
  else:
    dataset, labels = None, None

  return dataset, labels


def _merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = _make_arrays(valid_size, _IMAGE_SIZE)
  train_dataset, train_labels = _make_arrays(train_size, _IMAGE_SIZE)

  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_t, start_v = 0, 0
  end_t, end_v = tsize_per_class, vsize_per_class
  end_l = vsize_per_class + tsize_per_class

  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        np.random.shuffle(letter_set)

        if valid_dataset is not None:
          valid_letter_set = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter_set
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter_set = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter_set
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class

    except Exception as e:
      print ('Unable to process file %s due to %s' % (pickle_file, e))
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels

_TRAIN_SIZE = 200000
_VALID_SIZE = 10000
_TEST_SIZE = 10000


def _randomize_dataset(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation, :, :]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


def prepare_datasets(train_datasets, test_datasets):
  valid_dataset, valid_labels, train_dataset, train_labels = (
      _merge_datasets(train_datasets, _TRAIN_SIZE, _VALID_SIZE))
  _, _, test_dataset, test_labels = _merge_datasets(test_datasets, _TEST_SIZE)

  print('Training:', train_dataset.shape, train_labels.shape)
  print('Validation:', valid_dataset.shape, valid_labels.shape)
  print('Testing:', test_dataset.shape, test_labels.shape)
  train_dataset, train_labels = _randomize_dataset(train_dataset, train_labels)
  test_dataset, test_labels = _randomize_dataset(test_dataset, test_labels)
  valid_dataset, valid_labels = _randomize_dataset(valid_dataset, valid_labels)

  # Prepare object to pickle
  save = {
      'train_dataset': train_dataset,
      'train_labels': train_labels,
      'valid_dataset': valid_dataset,
      'valid_labels': valid_labels,
      'test_dataset': test_dataset,
      'test_labels': test_labels,
  }
  return save


def _save_processed_pickle(train_datasets, test_datasets):
  pickle_file_path = os.path.join(get_data_root(), get_pickle_file_name())
  if os.path.exists(pickle_file_path):
    print (
        'Pickle file %s already exists, not creating again' % get_pickle_file_name())
  else:
    try:
      save = prepare_datasets(train_datasets, test_datasets)
      f = open(pickle_file_path, 'wb')
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print ('Could not write to %s due to %s' % (pickle_file_path, e))
      raise

  return pickle_file_path


def prepare_data():
  train_filename = _maybe_download('notMNIST_large.tar.gz', 247336696)
  test_filename = _maybe_download('notMNIST_small.tar.gz', 8458043)

  num_classes = 10
  train_folders = _maybe_extract(train_filename, num_classes)
  test_folders = _maybe_extract(test_filename, num_classes)

  train_datasets = _maybe_pickle(train_folders, 45000)
  test_datasets = _maybe_pickle(test_folders, 1800)

  np.random.seed(133)
  pickle_file_path = _save_processed_pickle(train_datasets, test_datasets)
  statinfo = os.stat(pickle_file_path)
  print ('Pickle size %d' % statinfo.st_size)


if __name__ == '__main__':
  prepare_data()
