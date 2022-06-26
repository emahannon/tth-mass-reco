#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import keras

import keras.backend as K
import math
import matplotlib.pyplot as plt
import numpy as np


import tensorflow as tf

import time
import pandas as pd
import seaborn as sns

from keras import optimizers, metrics
from keras.layers import Dense, LayerNormalization, BatchNormalization, Dropout, GaussianNoise
from keras.models import load_model
from constants import column_labels_mass_reco


# In[2]:



# Modified from source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# Used to serve data from files to the neural network.
class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=32, n_features=67, shuffle=True, data_path="", scaler = "../scaler_params/particle_assignment_scaler_params.csv", write_to_file = False, standardize = True):
		'Initialization'
		self.n_features = n_features
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.write_to_file = write_to_file
		self.standardize = standardize
		self.data_path = data_path
		with open(scaler) as f:
			scaler_params = np.loadtxt(f, delimiter=",")
			self.scaler = scaler_params
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.ceil(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		# Generate data
		X, y, weights = self.__data_generation(list_IDs_temp)

		return X, y, weights

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.zeros((1,self.n_features),dtype=float)
		y = np.zeros((1,5),dtype=int)

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			# print(X.shape)
			# print((np.load(self.data_path +'/'+ str(int(ID)) + '.npy').shape))
			X = np.concatenate((X,np.load(self.data_path +'/'+ str(int(ID)) + '.npy')),axis=0)

			# Store class
			y = np.concatenate((y,self.labels[str(int(ID))]),axis=0)


		X = X[1:,:]
		y = y[1:,:]

		weights = np.reciprocal(X[:,-1])*200

		if self.write_to_file:
			X = np.concatenate((X[:,:-5],X[:,-4:]), axis = 1)   # Exclude weights.
		else:
			X = X[:,1:-5]
			if self.standardize:
				X = (X-self.scaler[0])/self.scaler[1]     # Standardize

		return X, y, weights

	def get_all(self):
		X = []
		y = []

		for i in range(self.__len__()):
			X_y = self.__getitem__(i)
			X += X_y[0].tolist()
			y += X_y[1].tolist()

		X = np.array(X)
		y = np.array(y)

		return X, y


# In[3]:


""" Loss function. """
# source: https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy
def create_weighted_binary_crossentropy(ones_weights,zeros_weights):

	def weighted_binary_crossentropy(y_true, y_pred):

		b_ce = K.binary_crossentropy(y_true, y_pred)
		weight_vector = y_true * ones_weights + (1. - y_true) * zeros_weights
		weighted_b_ce = weight_vector * b_ce

		return K.mean(weighted_b_ce)

	return weighted_binary_crossentropy


# In[4]:


""" Metrics """
# source: https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
def custom_f1(y_true, y_pred):
	def recall_m(y_true, y_pred):
		TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

		recall = TP / (Positives+K.epsilon())
		return recall


	def precision_m(y_true, y_pred):
		TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

		precision = TP / (Pred_Positives+K.epsilon())
		return precision

	precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

	return 2*((precision*recall)/(precision+recall+K.epsilon()))

# source: https://stackoverflow.com/questions/39895742/matthews-correlation-coefficient-with-keras
def matthews_correlation(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	numerator = (tp * tn - fp * fn)
	denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

	return numerator / (denominator + K.epsilon())

METRICS = [
	  keras.metrics.Precision(name='precision'),
	  keras.metrics.Recall(name='recall'),
	  keras.metrics.AUC(name='auc'),
	  keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
	  matthews_correlation,
	  custom_f1,
]


# In[5]:


# Load the trained model.
model = load_model("models/particle_assignment_model", custom_objects={'custom_f1':custom_f1, 'matthews_correlation':matthews_correlation, 'weighted_binary_crossentropy':create_weighted_binary_crossentropy})


# In[6]:


""" Load narrow selection ttH, ttZ data from files. """

with open("../data/particle_assignment_training_data/labels_dict_ttH.csv") as f:
	lines_ttH = f.readlines()
# with open("../data/particle_assignment_training_data/labels_dict_ttZ.csv") as f:
# 	lines_ttZ = f.readlines()

with open("../data/particle_assignment_training_data/train_ids_ttH.csv") as f:
	train_ids_ttH = np.loadtxt(f, delimiter=",")
# with open("../data/particle_assignment_training_data/train_ids_ttZ.csv") as f:
# 	train_ids_ttZ = np.loadtxt(f, delimiter=",")

with open("../data/particle_assignment_training_data/test_ids_ttH.csv") as f:
	test_ids_ttH = np.loadtxt(f, delimiter=",")
# with open("../data/particle_assignment_training_data/test_ids_ttZ.csv") as f:
# 	test_ids_ttZ = np.loadtxt(f, delimiter=",")

with open("../data/particle_assignment_training_data/val_ids_ttH.csv") as f:
	val_ids_ttH = np.loadtxt(f, delimiter=",")
# with open("../data/particle_assignment_training_data/val_ids_ttZ.csv") as f:
# 	val_ids_ttZ = np.loadtxt(f, delimiter=",")

# lines = np.concatenate((lines_ttH,lines_ttZ), axis = 0)
lines = lines_ttH

labels_dict = {}

for line in lines:
	row = np.fromstring(line, dtype=float, sep=',')
	key = str(int(row[0]))
	value = (row[1:]).reshape((int(len(row[1:])/5),5))
	labels_dict[key] = value


# In[11]:


""" The neural network produces probabilitise of each of the five positions being correctly assigned in a permutation of an event.
The permutation with the largest product of the respective probabilities is chosen.
The data of the chosen permutation is then saved to a file to be used in the mass reconstruction """

# Narrow ttH, ttZ.

# FIX THIS
# ids_list = [train_ids_ttH, val_ids_ttH, test_ids_ttH, train_ids_ttZ, val_ids_ttZ, test_ids_ttZ]
# file_name_endings = ["train_ttH","val_ttH","test_ttH","train_ttZ","val_ttZ","test_ttZ"]
# ids_list = [test_ids_ttH, test_ids_ttZ]
# file_name_endings = ["test_ttH","test_ttZ"]
ids_list = [train_ids_ttH, val_ids_ttH, test_ids_ttH]
file_name_endings = ["train_ttH","val_ttH","test_ttH"]

for ids,ending in zip(ids_list,file_name_endings):
	""" One generator for obtaining the probability vectors. Second generator to obtain the precise data we want to save to the file. """
	predict_generator = DataGenerator(ids, labels_dict, batch_size=128, shuffle=False, write_to_file=False, standardize=True, data_path = "../data/particle_assignment_training_data/data")
	write_to_file_generator = DataGenerator(ids, labels_dict, batch_size=128, shuffle=False, write_to_file=True, standardize=False, data_path = "../data/particle_assignment_training_data/data")

	test_ones_diff = 0
	test_samples_count = 0

	data = []
	all_best_ys = []
	all_best_possible_ys = []

	y_pred_best = []
	y_true_best = []
	best_matched = []

	for i in range(len(predict_generator)):
		X_test = predict_generator[i][0]
		y_test = predict_generator[i][1]

		X_to_file = write_to_file_generator[i][0]

		""" Get probability vectors. """
		preds = model.predict(X_test)

		X_y_preds = np.concatenate((X_to_file, y_test, preds),axis=1)

		ids = np.unique(X_to_file[:,0])

		X_to_file = X_to_file[:,1:]

		""" Each ID represents one event. """
		for id in ids:
			X_y_preds_all_combinations = np.array([row[1:] for row in X_y_preds if row[0] == id])
			X_all_combinations = X_y_preds_all_combinations[:,:X_to_file.shape[1]]
			y_all_combinations = X_y_preds_all_combinations[:,X_to_file.shape[1]:X_to_file.shape[1]+y_test.shape[1]]
			preds_all_combinations = X_y_preds_all_combinations[:,X_to_file.shape[1]+y_test.shape[1]:]

			product = np.product(preds_all_combinations, axis=1)

			""" Largest product yields the best (chosen) assignment. """
			best_pred = preds_all_combinations[np.argmax(product)]
			best_y = y_all_combinations[np.argmax(product)]
			best_possible_y = y_all_combinations[np.argmax(np.sum(y_all_combinations, axis = 1))]

			best_matched.append(np.sum(best_y) >= np.sum(best_possible_y))

			all_best_ys.append(best_y)
			all_best_possible_ys.append(best_possible_y)

			best_X = X_all_combinations[np.argmax(product)]

			""" Save data we will later write to file. """
			data += [best_X.tolist() + best_pred.tolist()]

	print(np.count_nonzero(best_matched)/len(best_matched))

	all_best_ys = np.mean(all_best_ys, axis=0)
	all_best_possible_ys = np.mean(all_best_possible_ys, axis=0)

	""" Besides saving the data we can also check the accuracy of each positions assignment (only for narrow selection). """

	f = open("../data/particle_assignment_training_data/particle_assignment_accuracy_narrow_selection_" + ending + ".csv", "w")
	writer = csv.writer(f)
	writer.writerow(all_best_ys)
	writer.writerow(all_best_possible_ys)
	f.close()

	""" Write to file. """

	f = open("../data/mass_reco/mass_reco_input_narrow_selection_" + ending + ".csv", "w")
	writer = csv.writer(f)
	writer.writerow(column_labels_mass_reco)
	writer.writerows(data)
	f.close()


# In[ ]:


""" Load wide selection ttH, ttZ data from files. """

with open("../data/particle_assignment_data_to_be_processed_ttH_ttZ/labels_dict_ttH.csv") as f:
	lines_ttH = f.readlines()
# with open("data/particle_assignment_data_to_be_processed_ttH_ttZ/labels_dict_ttZ.csv") as f:
# 	lines_ttZ = f.readlines()

with open("../data/particle_assignment_data_to_be_processed_ttH_ttZ/train_ids_ttH.csv") as f:
	train_ids_ttH = np.loadtxt(f, delimiter=",")
# with open("data/particle_assignment_data_to_be_processed_ttH_ttZ/train_ids_ttZ.csv") as f:
# 	train_ids_ttZ = np.loadtxt(f, delimiter=",")

with open("../data/particle_assignment_data_to_be_processed_ttH_ttZ/test_ids_ttH.csv") as f:
	test_ids_ttH = np.loadtxt(f, delimiter=",")
# with open("data/particle_assignment_data_to_be_processed_ttH_ttZ/test_ids_ttZ.csv") as f:
# 	test_ids_ttZ = np.loadtxt(f, delimiter=",")

with open("../data/particle_assignment_data_to_be_processed_ttH_ttZ/val_ids_ttH.csv") as f:
	val_ids_ttH = np.loadtxt(f, delimiter=",")
# with open("data/particle_assignment_data_to_be_processed_ttH_ttZ/val_ids_ttZ.csv") as f:
# 	val_ids_ttZ = np.loadtxt(f, delimiter=",")

#lines = np.concatenate((lines_ttH,lines_ttZ), axis = 0)
lines = lines_ttH

labels_dict = {}

for line in lines:
	row = np.fromstring(line, dtype=float, sep=',')
	key = str(int(row[0]))
	value = (row[1:]).reshape((int(len(row[1:])/5),5))
	labels_dict[key] = value


# In[ ]:


""" Same as for narrow selection, but this time the wide selection ttH and ttZ is used. """

# ids_list = [train_ids_ttH, val_ids_ttH, test_ids_ttH, train_ids_ttZ, val_ids_ttZ, test_ids_ttZ]
# file_name_endings = ["train_ttH", "val_ttH","test_ttH","train_ttZ","val_ttZ","test_ttZ"]
ids_list = [train_ids_ttH, val_ids_ttH, test_ids_ttH]
file_name_endings = ["train_ttH", "val_ttH","test_ttH"]


for ids,ending in zip(ids_list,file_name_endings):
	predict_generator = DataGenerator(ids, labels_dict, batch_size=128, shuffle=False, write_to_file=False, standardize=True, data_path = "../data/particle_assignment_data_to_be_processed_ttH_ttZ/data")
	write_to_file_generator = DataGenerator(ids, labels_dict, batch_size=128, shuffle=False, write_to_file=True, standardize=False, data_path = "../data/particle_assignment_data_to_be_processed_ttH_ttZ/data")

	test_ones_diff = 0
	test_samples_count = 0

	data = []
	all_best_ys = []
	all_best_possible_ys = []

	y_pred_best = []
	y_true_best = []
	for i in range(len(predict_generator)):
		X_test = predict_generator[i][0]
		y_test = predict_generator[i][1]

		X_to_file = write_to_file_generator[i][0]

		preds = model.predict(X_test)

		X_y_preds = np.concatenate((X_to_file, y_test, preds),axis=1)

		ids = np.unique(X_to_file[:,0])

		X_to_file = X_to_file[:,1:]

		for id in ids:
			X_y_preds_all_combinations = np.array([row[1:] for row in X_y_preds if row[0] == id])
			X_all_combinations = X_y_preds_all_combinations[:,:X_to_file.shape[1]]
			preds_all_combinations = X_y_preds_all_combinations[:,X_to_file.shape[1]+y_test.shape[1]:]

			product = np.product(preds_all_combinations, axis=1)

			best_pred = preds_all_combinations[np.argmax(product)]
			best_X = X_all_combinations[np.argmax(product)]

			data += [best_X.tolist() + best_pred.tolist()]

	f = open("../data/mass_reco/mass_reco_input_wide_selection_" + ending + ".csv", "w")
	writer = csv.writer(f)
	writer.writerow(column_labels_mass_reco)
	writer.writerows(data)
	f.close()


# In[ ]:

# COMMENTED OUT ALL ttW tt FROM HERE TO THE BOTTOM OF THE CODE
# """ Load wide selection ttW, tt data from files. """
#
# with open("../data/particle_assignment_data_to_be_processed_ttW_tt/labels_dict_ttW.csv") as f:
# 	lines_ttW = f.readlines()
# with open("../data/particle_assignment_data_to_be_processed_ttW_tt/labels_dict_tt.csv") as f:
# 	lines_tt = f.readlines()
#
# with open("../data/particle_assignment_data_to_be_processed_ttW_tt/train_ids_ttW.csv") as f:
# 	train_ids_ttW = np.loadtxt(f, delimiter=",")
# with open("../data/particle_assignment_data_to_be_processed_ttW_tt/train_ids_tt.csv") as f:
# 	train_ids_tt = np.loadtxt(f, delimiter=",")
#
# with open("../data/particle_assignment_data_to_be_processed_ttW_tt/test_ids_ttW.csv") as f:
# 	test_ids_ttW = np.loadtxt(f, delimiter=",")
# with open("../data/particle_assignment_data_to_be_processed_ttW_tt/test_ids_tt.csv") as f:
# 	test_ids_tt = np.loadtxt(f, delimiter=",")
#
# with open("../data/particle_assignment_data_to_be_processed_ttW_tt/val_ids_ttW.csv") as f:
# 	val_ids_ttW = np.loadtxt(f, delimiter=",")
# with open("../data/particle_assignment_data_to_be_processed_ttW_tt/val_ids_tt.csv") as f:
# 	val_ids_tt = np.loadtxt(f, delimiter=",")
#
# lines = np.concatenate((lines_ttW,lines_tt), axis = 0)
#
# labels_dict = {}
#
# for line in lines:
# 	row = np.fromstring(line, dtype=float, sep=',')
# 	key = str(int(row[0]))
# 	value = (row[1:]).reshape((int(len(row[1:])/5),5))
# 	labels_dict[key] = value
#
#
# # In[ ]:
#
#
# """ Same as for narrow selection, but this time the wide selection ttW and tt is used. """
#
# ids_list = [train_ids_ttW, val_ids_ttW, test_ids_ttW, train_ids_tt, val_ids_tt, test_ids_tt]
# file_name_endings = ["train_ttW","val_ttW","test_ttW","train_tt","val_tt","test_tt"]
#
#
# for ids,ending in zip(ids_list,file_name_endings):
# 	predict_generator = DataGenerator(ids, labels_dict, batch_size=128, shuffle=False, write_to_file=False, standardize=True, data_path="data/particle_assignment_data_to_be_processed_ttW_tt/data")
# 	write_to_file_generator = DataGenerator(ids, labels_dict, batch_size=128, shuffle=False, write_to_file=True, standardize=False, data_path="data/particle_assignment_data_to_be_processed_ttW_tt/data")
#
# 	test_ones_diff = 0
# 	test_samples_count = 0
#
# 	data = []
# 	all_best_ys = []
# 	all_best_possible_ys = []
#
# 	y_pred_best = []
# 	y_true_best = []
# 	for i in range(len(predict_generator)):
# 		X_test = predict_generator[i][0]
# 		y_test = predict_generator[i][1]
#
# 		X_to_file = write_to_file_generator[i][0]
#
# 		preds = model.predict(X_test)
#
# 		X_y_preds = np.concatenate((X_to_file, y_test, preds),axis=1)
#
# 		ids = np.unique(X_to_file[:,0])
#
# 		X_to_file = X_to_file[:,1:]
#
# 		for id in ids:
# 			X_y_preds_all_combinations = np.array([row[1:] for row in X_y_preds if row[0] == id])
# 			X_all_combinations = X_y_preds_all_combinations[:,:X_to_file.shape[1]]
# 			preds_all_combinations = X_y_preds_all_combinations[:,X_to_file.shape[1]+y_test.shape[1]:]
#
# 			product = np.product(preds_all_combinations, axis=1)
#
# 			best_pred = preds_all_combinations[np.argmax(product)]
# 			best_X = X_all_combinations[np.argmax(product)]
#
# 			data += [best_X.tolist() + best_pred.tolist()]
#
# 	f = open("../data/mass_reco/mass_reco_input_wide_selection_" + ending + ".csv", "w")
# 	writer = csv.writer(f)
# 	writer.writerow(column_labels_mass_reco)
# 	writer.writerows(data)
# 	f.close()
