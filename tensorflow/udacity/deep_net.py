from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import download_extract_process
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Or, you could use the file that has all the images.
pickle_file = 'notMNIST_unique.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

_IMAGE_SIZE = 28
_NUM_LABELS = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, _IMAGE_SIZE * _IMAGE_SIZE)).astype(np.float32)
  labels = (np.arange(_NUM_LABELS) == labels[:, None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print ('Training set', train_dataset.shape, train_labels.shape)
print ('Training set', valid_dataset.shape, valid_labels.shape)
print ('Training set', test_dataset.shape, test_labels.shape)

# HYPER PARAMETERS
_TRAIN_SUBSET = 100000
_NUM_HIDDEN_NEURONS_L1 = 100
_NUM_HIDDEN_NEURONS_L2 = 100
_L2_BETA = 0.0001
_LEARNING_RATE = 0.1
_DROPOUT_KEEP_PROBABILITY = 0.8
_SGD_BATCH_SIZE = 5000
_NUM_STEPS = 3000

graph = tf.Graph()
with graph.as_default():
  tf_train_dataset = tf.placeholder(tf.float32, [None, _IMAGE_SIZE * _IMAGE_SIZE])
  tf_train_labels = tf.placeholder(tf.float32, [None, _NUM_LABELS])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  tf_keep_probability = tf.placeholder(tf.float32)
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(_LEARNING_RATE, global_step, 100000, 0.96, staircase=True)

  # Hidden layer
  hidden_weights = tf.Variable(tf.truncated_normal([_IMAGE_SIZE * _IMAGE_SIZE, _NUM_HIDDEN_NEURONS_L1]))
  hidden_biases = tf.Variable(tf.zeros([_NUM_HIDDEN_NEURONS_L1]))
  dropped_hidden_weights = tf.nn.dropout(hidden_weights, tf_keep_probability)
  hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, dropped_hidden_weights) + hidden_biases)

  # Hidden layer 2
  hidden_weights_2 = tf.Variable(tf.truncated_normal([_NUM_HIDDEN_NEURONS_L1, _NUM_HIDDEN_NEURONS_L2]))
  hidden_biases_2 = tf.Variable(tf.zeros([_NUM_HIDDEN_NEURONS_L2]))
  dropped_hidden_weights_2 = tf.nn.dropout(hidden_weights_2, tf_keep_probability)
  hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer, dropped_hidden_weights_2) + hidden_biases_2)

  # Output layer
  output_weights = tf.Variable(tf.truncated_normal([_NUM_HIDDEN_NEURONS_L2, _NUM_LABELS]))
  output_biases = tf.Variable(tf.zeros([_NUM_LABELS]))
  dropped_output_weights = tf.nn.dropout(output_weights, tf_keep_probability)
  output_layer = tf.matmul(hidden_layer_2, dropped_output_weights) + output_biases

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=tf_train_labels))

  # Add L2 regularization penalty
  loss += (_L2_BETA * tf.nn.l2_loss(dropped_hidden_weights)
           + _L2_BETA * tf.nn.l2_loss(dropped_hidden_weights_2)
           + _L2_BETA * tf.nn.l2_loss(dropped_output_weights))

  optimizer = tf.train.GradientDescentOptimizer(_LEARNING_RATE).minimize(loss, global_step=global_step)

  # Predictions
  train_prediction = tf.nn.softmax(output_layer)

  valid_relu = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
  valid_relu_2 = tf.nn.relu(tf.matmul(valid_relu, hidden_weights_2) + hidden_biases_2)
  valid_prediction = tf.nn.softmax(tf.matmul(valid_relu_2, output_weights) + output_biases)

  test_relu = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
  test_relu_2 = tf.nn.relu(tf.matmul(test_relu, hidden_weights_2) + hidden_biases_2)
  test_prediction = tf.nn.softmax(tf.matmul(test_relu_2, output_weights) + output_biases)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def train_sgd_with_l2_regularization():
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print ('Initialized')
    results = {
        'loss': [],
        'training_accuracy': [],
        'validation_accuracy': [],
        'test_accuracy': None
    }
    for step in range(_NUM_STEPS):
      # Get a random batch of training data and labels.
      # Respect the train subset that we are training on: _TRAIN_SUBSET
      random_indices = np.arange(_TRAIN_SUBSET)
      np.random.shuffle(random_indices)
      random_indices = random_indices[:_SGD_BATCH_SIZE]
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict={
          tf_train_dataset: train_dataset[random_indices, :],
          tf_train_labels: train_labels[random_indices, :],
          tf_keep_probability: _DROPOUT_KEEP_PROBABILITY
      })

      training_accuracy = accuracy(predictions, train_labels[random_indices, :])
      validation_accuracy = accuracy(valid_prediction.eval(), valid_labels)

      results['loss'].append(l)
      results['training_accuracy'].append(training_accuracy)
      results['validation_accuracy'].append(validation_accuracy)

      if step % 100 == 0:
        print ('At step %d' % step)
        print ('Loss %f' % l)
        print ('Training accuracy: %d' % training_accuracy)
        print ('Validation accuracy: %s' % validation_accuracy)

    results['test_accuracy'] = accuracy(test_prediction.eval(), test_labels)
    print ('Test accuracy: %s' % results['test_accuracy'])
    return results

def overfit_model_with_small_batch():
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print ('Initialized')
    batch_size = 100
    random_indices = np.arange(_TRAIN_SUBSET)
    np.random.shuffle(random_indices)
    random_indices = random_indices[:batch_size]

    # Get a batch for training, we'll try to overfit the model on this batch.
    small_training_batch = train_dataset[random_indices, :]
    small_training_labels = train_labels[random_indices, :]

    results = {
        'loss': [],
        'training_accuracy': [],
        'validation_accuracy': [],
        'test_accuracy': None
    }

    for step in range(_NUM_STEPS):
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict={
          tf_train_dataset: small_training_batch,
          tf_train_labels: small_training_labels,
          tf_keep_probability: _DROPOUT_KEEP_PROBABILITY
      })

      training_accuracy = accuracy(predictions, small_training_labels)
      validation_accuracy = accuracy(valid_prediction.eval(), valid_labels)

      results['loss'].append(l)
      results['training_accuracy'].append(training_accuracy)
      results['validation_accuracy'].append(validation_accuracy)

      if step % 100 == 0:
        print ('At step %d' % step)
        print ('Loss %f' % l)
        print ('Training accuracy: %d' % training_accuracy)
        print ('Validation accuracy: %s' % validation_accuracy)

    results['test_accuracy'] = accuracy(test_prediction.eval(), test_labels)
    print ('Test accuracy: %s' % results['test_accuracy'])
    return results

def plot_results(results):
  gs = gridspec.GridSpec(3, 1)
  fig = plt.figure()
  ax1 = fig.add_subplot(gs[0])
  ax1.plot(results['loss'])
  ax1.set_ylabel('loss')

  ax2 = fig.add_subplot(gs[1])
  ax2.plot(results['training_accuracy'])
  ax2.set_ylabel('training accuracy')

  ax3 = fig.add_subplot(gs[2])
  ax3.plot(results['validation_accuracy'])
  ax3.set_ylabel('validation accuracy')
  plt.show()

# results = overfit_model_with_small_batch()
results = train_sgd_with_l2_regularization()
plot_results(results)








