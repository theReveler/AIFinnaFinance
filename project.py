import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

def batch_generator(batch_size, sequence_length):
	"""
	Generator function for creating random batches of training-data.
	"""

	# Infinite Loop
	while True:
		# Allocate a new array for the batch of input-signals.
		x_shape = (batch_size, sequence_length, num_x_signals)
		x_batch = np.zeros(shape=x_shape, dtype=np.float16)

		# Allocate a new array for the batch of output-signals.
		y_shape = (batch_size, sequence_length, num_y_signals)
		y_batch = np.zeros(shape=y_shape, dtype=np.float16)

		# Fill the batch with random sequences of data.
		for i in range(batch_size):
			# Get a random start-index.
			# This points somewhere into the training-data.
			idx = np.random.randint(num_train - sequence_length)

			# Copy the sequences of data starting at this index.
			x_batch[i] = x_train_scaled[idx:idx+sequence_length]
			y_batch[i] = y_train_scaled[idx:idx+sequence_length]

		yield (x_batch, y_batch)

def loss_mse_warmup(y_true, y_pred):
	"""Calculate the Mean Absolute Percentage Error between y_true and y_pred,
	but ignore the beginning "warmup" part of the sequences.

	y_true is the desired output.
	y_pred is the model's output.
	"""

	# The shape of both input tensors are:
	# [batch_size, sequence_length, num_y_signals].

	# Ignore the "warmup" parts of the sequences by taking slices of the tensors
	y_true_slice = y_true[:, warmup_steps:, :]
	y_pred_slice = y_pred[:, warmup_steps:, :]

	# These sliced tensors both have this shape:
	# [batch_size, sequence_length - warmup_steps, num_y_signals]

	# Caluculate the MAPE loss for each value in these tensors.
	# This outputs a 3-rank tensor of the same shape.
	loss = tf.losses.mean_squared_error(labels=y_true_slice, predictions=y_pred_slice)
	loss_mean = tf.reduce_mean(loss)
	return loss_mean

def plot_comparison(start_idx, train=True):
	"""
	Plot the predicted and true output-signals.

	:param start_idx: Start-index for the time-series.
	:param length: Sequence-length to process and plot.
	:param train: Boolean whether to use training- or test-set.
	"""

	if train:
		x = x_train_scaled
		y_true = y_train
	else:
		x = x_test_scaled
		y_true = y_test

	x = x[start_idx:]
	y_true = y_true[start_idx:]

	x = np.expand_dims(x, axis=0)
	y_pred = model.predict(x)

	if train:
		y_pred_rescaled = y_train_scaler.inverse_transform(y_pred[0])
	else:
		y_pred_rescaled = y_test_scaler.inverse_transform(y_pred[0])

	signal_pred = y_pred_rescaled[:, 0]
	signal_true = y_true[:, 0]

	plt.figure(figsize=(15, 5))
	plt.plot(signal_true, label='true')
	plt.plot(signal_pred, label='pred')

	# Plot grey box for warmup-period
	p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

	plt.ylabel('Close')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	# Import data
	path = os.path.dirname(os.path.realpath(__file__)) + '\\AMZN.csv'
	df = pd.read_csv(path, index_col=False, header=0)

	df['Date'] = pd.to_datetime(df['Date'])
	df = df.set_index('Date')
	df.plot(y=3)
	plt.show()

	# Split data into training and test
	shift_days = 1
	df_targets = df['Close'].shift(-shift_days)
	x_data = df.values[0:-shift_days]
	y_data = df_targets.values[:-shift_days].reshape(-1, 1)
	num_data = len(x_data)
	num_x_signals = x_data.shape[1]
	num_y_signals = y_data.shape[1]
	train_split = 0.8
	num_train = int(train_split * num_data)
	num_test = num_data - num_train
	x_train = x_data[0:num_train]
	x_test = x_data[num_train:]
	y_train = y_data[0:num_train]
	y_test = y_data[num_train:]
	#x_batches = x_train.reshape(-1, 25, 1)
	#y_batches = y_train.reshape(-1, 25, 1)

	# Scale data
	x_train_scaler = MinMaxScaler()
	x_test_scaler = MinMaxScaler()
	y_train_scaler = MinMaxScaler()
	y_test_scaler = MinMaxScaler()
	x_train_scaled = x_train_scaler.fit_transform(x_train)
	x_test_scaled = x_test_scaler.fit_transform(x_test)
	y_train_scaled = y_train_scaler.fit_transform(y_train)
	y_test_scaled = y_test_scaler.fit_transform(y_test)


	# Data generator
	sequence_length = 15
	batch_size = 281
	generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)

	# Validation set
	validation_data = (np.expand_dims(x_test_scaled, axis=0),
	                   np.expand_dims(y_test_scaled, axis=0))

	# Create the RNN
	model = Sequential()
	model.add(GRU(units=512, return_sequences=True, input_shape=(None, num_x_signals,)))
	model.add(Dense(num_y_signals, activation='relu'))	# Use sigmoid/relu

	# Loss function warmup-period
	warmup_steps = 0

	# Compile model
	optimizer = RMSprop(lr=1e-3)
	model.compile(loss=loss_mse_warmup, optimizer=optimizer)

	# Callback functions
	path_checkpoint = 'checkpoint.keras'
	callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
	                                      monitor='val_loss',
										  verbose=1,
										  save_weights_only=True,
										  save_best_only=True)
	callback_early_stopping = EarlyStopping(monitor='val_loss',
	                                        patience=5, verbose=1)
	callback_tensorboard = TensorBoard(log_dir='./',
	                                   histogram_freq=0,
									   write_graph=True,
									   write_images=True)
	callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
	                                       factor=0.1,
										   min_lr=1e-4,
										   patience=0,
										   verbose=1)
	callbacks = [callback_early_stopping,
	             callback_checkpoint,
				 callback_tensorboard,
				 callback_reduce_lr]

	# Train the RNN
	model.fit_generator(generator=generator,
	                    epochs=20,
						steps_per_epoch=100,
						validation_data=validation_data,
						callbacks=callbacks)

	# Load last saved checkpoint, which has the best performance on the test-set
	try:
		model.load_weights(path_checkpoint)
	except Exception as error:
		print("Error trying to load checkpoing.")
		print(error)

	# Performance on Test-Set
	result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
	                        y=np.expand_dims(y_test_scaled, axis=0))
	print("Loss (test-set):", result)

	# Plot results
	plot_comparison(start_idx=0, train=True)
	plot_comparison(start_idx=0, train=False)
	
	'''
	TS = np.array(df['close'])
	num_periods = 30
	f_horizon = 1	#forecast horizon, one period into the future
	x_data = TS[:(len(TS)-(len(TS) % num_periods))]
	x_batches = x_data.reshape(-1, 30, 1)
	y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
	y_batches = y_data.reshape(-1, 30, 1)

	#X_test, Y_test = test_data(TS, f_horizon, num_periods)

	#TensorFlow
	tf.reset_default_graph()
	num_periods = 25
	inputs = 1
	hidden = 100
	output = 1

	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, [None, num_periods, inputs], name="x-input")
		y = tf.placeholder(tf.float32, [None, num_periods, output], name="y-input")

	basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
	rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

	learning_rate = 0.001

	stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
	stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
	outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

	with tf.name_scope('MAPE'):
		loss = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(outputs, y), y)))
	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		training_op = optimizer.minimize(loss)

	init = tf.global_variables_initializer()

	#Training
	epochs = 100000
	logs_path = os.path.dirname(os.path.realpath(__file__))
	tf.summary.scalar("mape", loss)
	summary_op = tf.summary.merge_all()

	with tf.Session() as sess:
		init.run()

		writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		for ep in range(epochs):
			_, summary = sess.run([training_op, summary_op], feed_dict={X: x_batches, y: y_batches})
			writer.add_summary(summary, ep)
			if ep % 1000 == 0:
				mape = loss.eval(feed_dict={X: x_batches, y: y_batches})
				print(ep, "\tMAPE:", mape)

		#y_pred = sess.run(outputs, feed_dict={X: X_test})
		#print(y_pred)
		y_pred = sess.run(outputs, feed_dict={X: x_batches})

	plt.title("Forecast vs Actual", fontsize=14)
	#plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Actual")
	#plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forecast")
	plt.plot(pd.Series(np.ravel(y_batches)), "b-", markersize=0.5, label="Actual")
	plt.plot(pd.Series(np.ravel(y_pred)), "r-", markersize=0.5, label="Forecast")
	plt.legend(loc="upper left")
	plt.xlabel("Time Periods")

	plt.show()
	'''
