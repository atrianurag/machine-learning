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


# Input parameters
image_size = 28
num_labels = 10
num_channels = 1

# Hyper parameters
batch_size = 16
patch_size = 10
depth_1 = 16
depth_2 = 32
num_hidden = 64
step_size = 0.01
num_steps = 1001
_L2_BETA = 0.001

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print ('Training set', train_dataset.shape, train_labels.shape)
print ('Valid set', valid_dataset.shape, valid_labels.shape)
print ('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

graph = tf.Graph()

with graph.as_default():
  w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth_1], stddev=0.1))
  w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_1, depth_2], stddev=0.1))
  w3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth_2, num_hidden], stddev=0.1))
  w4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
  weights = [w1, w2, w3, w4]

  b1 = tf.Variable(tf.random_normal([depth_1]))
  b2 = tf.Variable(tf.random_normal([depth_2]))
  b3 = tf.Variable(tf.random_normal([num_hidden]))
  b4 = tf.Variable(tf.random_normal([num_labels]))

  tf_train_dataset = tf.placeholder(
      tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(
      tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  def model_with_pooling(data):
    # First convolution with depth = depth_1
    conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pool + b1)

    # Second convolution with depth = depth_2
    conv = tf.nn.conv2d(hidden, w2, [1, 1, 1, 1], padding='SAME')
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pool + b2)

    # First hidden layer with size = num_hidden
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, w3) + b3)

    # Final layer with size = num_labels
    return tf.nn.relu(tf.matmul(hidden, w4) + b4)

  logits = model_with_pooling(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  for w in weights:
    loss += _L2_BETA * tf.nn.l2_loss(w)

  optimizer = tf.train.GradientDescentOptimizer(step_size).minimize(loss)

  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model_with_pooling(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model_with_pooling(tf_test_dataset))

def train_conv_net():
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print ('initialized')
    results = {
        'loss': [],
        'training_accuracy': [],
        'validation_accuracy': [],
        'test_accuracy': None
    }

    for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
      _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)

      if (step % 50 == 0):
        training_accuracy = accuracy(predictions, batch_labels)
        validation_accuracy = accuracy(valid_prediction.eval(), valid_labels)
        results['loss'].append(l)
        results['training_accuracy'].append(training_accuracy)
        results['validation_accuracy'].append(validation_accuracy)
        print ('Minibatch loss at step %d: %f' % (step, l))
        print ('Minibatch accuracy at: %.1f%%' % training_accuracy)
        print ('Validation accuracy at: %.1f%%' % validation_accuracy)

    results['test_accuracy'] = accuracy(test_prediction.eval(), test_labels)
    print ('Test accuracy: %.1f%%' % results['test_accuracy'])
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

results = train_conv_net()
plot_results(results)














