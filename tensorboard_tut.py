# From https://www.datacamp.com/community/tutorials/tensorboard-tutorial

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def accuracy(predictions, labels):
	return np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) * 100.0 / labels.shape[0]


batch_size = 80
# layer_ids = ['hidden1', 'hidden2', 'hidden3', 'hidden4', 'hidden5', 'out']
# layer_sizes = [784, 500, 400, 300, 200, 100, 10]
layer_ids = ['hidden1', 'hidden2', 'out']
layer_sizes = [784, 450, 250,  10]

tf.reset_default_graph()

# Inputs and Labels
train_inputs = tf.placeholder(tf.float32, shape=[batch_size, layer_sizes[0]], name='train_inputs')
train_labels = tf.placeholder(tf.float32, shape=[batch_size, layer_sizes[-1]], name='train_labels')

# Weight and Bias definitions
for idx, lid in enumerate(layer_ids):
	with tf.variable_scope(lid):
		w = tf.get_variable('weights', shape=[layer_sizes[idx], layer_sizes[idx + 1]],
							initializer=tf.truncated_normal_initializer(stddev=0.05))
		b = tf.get_variable('bias', shape=[layer_sizes[idx + 1]], initializer=tf.random_uniform_initializer(-0.1, 0.1))

# Calculating logits
h = train_inputs
for lid in layer_ids:
	with tf.variable_scope(lid, reuse=True):
		w, b = tf.get_variable('weights'), tf.get_variable('bias')
		if lid != 'out':
			h = tf.nn.relu(tf.matmul(h, w) + b, name=lid + '_output')
		else:
			h = tf.nn.xw_plus_b(h, w, b, name=lid + '_output')

tf_predictions = tf.nn.softmax(h, name='predictions')

# Calculating loss
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_labels, logits=h), name='loss')

# Optimizer
tf_learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
optimizer = tf.train.MomentumOptimizer(tf_learning_rate, momentum=0.9)
grads_and_vars = optimizer.compute_gradients(tf_loss)
tf_loss_minimize = optimizer.minimize(tf_loss)

# Name scope allows you to group various summaries_tut together
# Summaries having the same name_scope will be displayed on the same row
with tf.name_scope('performance'):
	# Summaries need to be displayed
	# Whenever you need to record the loss, feed the mean loss to this placeholder
	tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
	# Create a scalar summary object for the loss so it can be displayed
	tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

# Whenever you need to record the loss, feed the mean test accuracy to this placeholder
tf_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
# Create a scalar summary object for the accuracy so it can be displayed
tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

# Gradient norm summary
for g, v in grads_and_vars:
	if layer_ids[-2] in v.name and 'weights' in v.name:
		with tf.name_scope('gradients'):
			tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g ** 2))
			tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
			break

# Merge all summaries_tut together
performance_summaries = tf.summary.merge([tf_loss_summary, tf_accuracy_summary])

image_size = 28
n_channels = 1
n_classes = 10
n_train = 55000
n_valid = 5000
n_test = 1000
n_epochs = 30

# config = tf.ConfigProto(allow_soft_placements=True)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

# session = tf.InteractiveSession(config=config)
session = tf.InteractiveSession()

summary_name = '2-layers'
if not os.path.exists('summaries_tut'):
	os.mkdir('summaries_tut')
if not os.path.exists(os.path.join('summaries_tut', summary_name)):
	os.mkdir(os.path.join('summaries_tut', summary_name))

summ_writer = tf.summary.FileWriter(os.path.join('summaries_tut', summary_name), session.graph)

tf.global_variables_initializer().run()

accuracy_per_epoch = []
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

for epoch in range(n_epochs):
	loss_per_epoch = []
	for i in range(n_train // batch_size):
		# =================================== Training for one step ========================================
		batch = mnist_data.train.next_batch(batch_size)  # Get one batch of training data
		if i == 0:
			# Only for the first epoch, get the summary data
			# Otherwise, it can clutter the visualization
			l, _, gn_summ = session.run([tf_loss, tf_loss_minimize, tf_gradnorm_summary],
										feed_dict={train_inputs: batch[0].reshape(batch_size, image_size * image_size),
												   train_labels: batch[1],
												   tf_learning_rate: 0.0001})
			summ_writer.add_summary(gn_summ, epoch)
		else:
			# Optimize with training data
			l, _= session.run([tf_loss, tf_loss_minimize], feed_dict={train_inputs: batch[0].reshape(batch_size, image_size * image_size), train_labels: batch[1], tf_learning_rate: 0.0001})
		loss_per_epoch.append(l)

	avg_loss = np.mean(loss_per_epoch)
	print(f"Average loss in epoch {epoch}: {avg_loss:.2f}")

	# ====================== Calculate the Validation Accuracy ==========================
	valid_accuracy_per_epoch = []
	for i in range(n_valid // batch_size):
		valid_images, valid_labels = mnist_data.validation.next_batch(batch_size)
		valid_batch_predictions = session.run(
			tf_predictions, feed_dict={train_inputs: valid_images.reshape(batch_size, image_size * image_size)})
		valid_accuracy_per_epoch.append(accuracy(valid_batch_predictions, valid_labels))

	mean_v_acc = np.mean(valid_accuracy_per_epoch)
	print(f"\tAverage Valid Accuracy in epoch {epoch}: {mean_v_acc:.5f}")

	# ===================== Calculate the Test Accuracy ===============================
	accuracy_per_epoch = []
	for i in range(n_test // batch_size):
		test_images, test_labels = mnist_data.test.next_batch(batch_size)
		test_batch_predictions = session.run(
			tf_predictions, feed_dict={train_inputs: test_images.reshape(batch_size, image_size * image_size)}
		)
		accuracy_per_epoch.append(accuracy(test_batch_predictions, test_labels))

	avg_test_accuracy = np.mean(accuracy_per_epoch)
	print(f"\tAverage Test Accuracy in epoch {epoch}: {avg_test_accuracy:.5f}")

	# Execute the summaries_tut defined above
	summ = session.run(performance_summaries, feed_dict={tf_loss_ph: avg_loss, tf_accuracy_ph: avg_test_accuracy})

	# Write the obtained summaries_tut to the file, so it can be displayed in the TensorBoard
	summ_writer.add_summary(summ, epoch)

session.close()
