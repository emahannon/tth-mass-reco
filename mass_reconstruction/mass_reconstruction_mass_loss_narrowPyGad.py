# %%
# In this model we remove Adam's structure and replace it with a tf sequential
# 	model of the same structure.
import collections
import csv
import keras
import keras.backend as K
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import sys
import tensorflow as tf
import time
import shap
import ROOT as root
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from keras import optimizers
from keras.regularizers import l2

from keras.layers import Dense, LayerNormalization, BatchNormalization, Dropout, GaussianNoise, Activation, Add
from keras import activations

from tensorflow.keras import Sequential


# imports for pygad
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
import math
import multiprocessing
import concurrent.futures
import concurrent
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor 
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import operator
import time
import tensorflow.keras
import pygad.kerasga
import numpy
import pygad


from ROOT import TFile, TLorentzVector

from constants import *

# %%
""" Data augmentation. """
def random_rotation(X,deltas_phi,num_vectors,met):

	for i in range(num_vectors):

		R = np.sqrt((X[:,4*i+0]**2 + X[:,4*i+1]**2))
		old_phis = np.arctan2(X[:,4*i+1],X[:,4*i+0])
		new_phis = old_phis + deltas_phi

		angles_sin = np.sin(new_phis)
		angles_cos = np.cos(new_phis)

		X[:,4*i+0] = angles_cos * R
		X[:,4*i+1] = angles_sin * R

	if met:
		return random_rotation_MET(X, deltas_phi)
	else:
		return X


def random_rotation_MET(X, deltas_phi):

	R = np.sqrt((X[:,-11]**2 + X[:,-10]**2))
	old_phis = np.arctan2(X[:,-10],(X[:,-11]))
	new_phis = old_phis + deltas_phi

	angles_sin = np.sin(new_phis)
	angles_cos = np.cos(new_phis)

	X[:,-11] = angles_cos * R
	X[:,-10] = angles_sin * R

	return X


def data_rotation(X, num_vectors_X = 11, met = True):

	deltas_phi = np.random.rand(X.shape[0])*2*math.pi
	X = random_rotation(X,deltas_phi,num_vectors_X,met)

	return X

""" Data generator for serving the NN data. """
# modified from source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	# commented out vv bc we have the same files but with the btags part of the name removed, they are the same thing but the naming just got messed up
	# def __init__(self, X, y, scaler = ["../scaler_params/X_scaler_mass_reco_narrow_btags.csv","scaler_params/y_scaler_higgs_masses_narrow_btags.csv"], batch_size=32, n_features=76, shuffle=True, augmentation = True, ceil = False):
	def __init__(self, X, y, scaler = ["../scaler_params/X_scaler_mass_reco_narrow.csv", "../scaler_params/y_scaler_higgs_masses_narrow.csv"], batch_size=32, n_features=67, shuffle=True, augmentation = True, ceil = False):
		self.n_features = n_features
		self.batch_size = batch_size
		self.X = X
		self.y = y
		self.shuffle = shuffle
		self.augmentation = augmentation
		self.ceil = ceil
		self.indexes = np.arange(len(X))

		with open(scaler[0]) as f:
			scaler_params = np.loadtxt(f, delimiter=",")
			self.X_scaler = scaler_params[0:2,:]

		with open(scaler[1]) as f:
			scaler_params = np.loadtxt(f, delimiter=",")
			self.y_scaler = scaler_params[0:2]

		self.on_epoch_end()
		print("end on epoch end")

	def __len__(self):
		'Denotes the number of batches per epoch'

		if self.ceil:
			return int(np.ceil(len(self.X) / self.batch_size))
		return int(np.floor(len(self.X) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y, weights = self.__data_generation(indexes)
		return X, y, weights

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

		# is indexes a list of all the cols of the feature names we want to use?
		X = self.X[indexes]

		weights = X[:,-1]
		X = X[:,:-1]

		y = self.y[indexes]

		if self.augmentation:
			X = data_rotation(X, num_vectors_X = 11, met = True)

		# THESE LINES SHOULD ONLY BE COMMENTED OUT TO TRAIN THE SCALAR
		X = (X-self.X_scaler[0])/self.X_scaler[1]     # Standardize
		y = (y-self.y_scaler[0])/self.y_scaler[1]     # Standardize

		return X, y, weights

	def get_all(self):
		'Get all data - all batches.'
		X = []
		y = []

		for i in range(self.__len__()):
			X_y = self.__getitem__(i)
			X += X_y[0].tolist()
			y += X_y[1].tolist()

		X = np.array(X)
		y = np.array(y)
		return X, y

# %%
# Load data.

def add_weights(tth,ttz):
	""" Class weights, because of the imbalance between the productions' number of events. """
	weight_tth = 1/(2*len(tth)/(len(ttz)+len(tth)))
	weight_ttz = 1/(2*len(ttz)/(len(ttz)+len(tth)))

	ones = np.ones((tth.shape[0],1))
	tth = np.concatenate((tth, weight_tth*ones),axis=1)

	ones = np.ones((ttz.shape[0],1))
	ttz = np.concatenate((ttz, weight_ttz*ones),axis=1)

	return tth,ttz

# def add_weights(tth):
# 	""" Class weights, because of the imbalance between the productions' number of events. """
# 	weight_tth = 1/(2*len(tth)/(len(tth)))
#
#
# 	ones = np.ones((tth.shape[0],1))
# 	tth = np.concatenate((tth, weight_tth*ones),axis=1)
#
# 	return tth

with open("../data/mass_reco/mass_reco_input_narrow_selection_train_ttH.csv") as f:
	X_y_train_ttH = np.loadtxt(f, delimiter=",", skiprows=1)
with open("../data/mass_reco/mass_reco_input_narrow_selection_train_ttZ.csv") as f:
	X_y_train_ttZ = np.loadtxt(f, delimiter=",", skiprows=1)

with open("../data/mass_reco/mass_reco_input_narrow_selection_test_ttH.csv") as f:
	X_y_test_ttH = np.loadtxt(f, delimiter=",", skiprows=1)
with open("../data/mass_reco/mass_reco_input_narrow_selection_test_ttZ.csv") as f:
	X_y_test_ttZ = np.loadtxt(f, delimiter=",", skiprows=1)

with open("../data/mass_reco/mass_reco_input_narrow_selection_val_ttH.csv") as f:
	X_y_val_ttH = np.loadtxt(f, delimiter=",", skiprows=1)
with open("../data/mass_reco/mass_reco_input_narrow_selection_val_ttZ.csv") as f:
	X_y_val_ttZ = np.loadtxt(f, delimiter=",", skiprows=1)


feature_names = pd.read_csv("../data/mass_reco/mass_reco_input_narrow_selection_test_ttH.csv", nrows=1)
feature_names = list(feature_names.columns)


# removing features based on permutation feature importance
# here, we only remove features that are 0 or close to 0
# EVERY TIME WE REMOVE/ADD FEATURES WE NEED TO RETRAIN THE SCALER
bad_features = ['top W q2 pz', 'visible main E', 'hadr tau E', 'top W q2 E', 'visible antitop E', 'top W q1 E', 'antitop b E', 'top b E', 'antitop lepton E', 'top E', 'main lepton E', 'true main px', 'true main py', 'top W E', 'antitop b pz', 'anti top b and main lep delta R', 'antitop b and anti top lep delta R', 'top b pz', 'q1 and q2 delta R', 'top pz', 'top py', 'top and antitop delta R', 'antitop lepton pz']
temp = set(bad_features)
index_bad_features = [i for i, val in enumerate(feature_names) if val in temp]
feature_names = np.delete(feature_names, index_bad_features)

print(index_bad_features)
print(feature_names)


X_y_train_ttH = np.delete(X_y_train_ttH, index_bad_features, axis=1)
X_y_train_ttZ = np.delete(X_y_train_ttZ, index_bad_features, axis=1)

X_y_test_ttH = np.delete(X_y_test_ttH, index_bad_features, axis=1)
X_y_test_ttZ = np.delete(X_y_test_ttZ, index_bad_features, axis=1)

X_y_val_ttH = np.delete(X_y_val_ttH, index_bad_features, axis=1)
X_y_val_ttZ = np.delete(X_y_val_ttZ, index_bad_features, axis=1)





X_y_train_ttH, X_y_train_ttZ = add_weights(X_y_train_ttH, X_y_train_ttZ)    # Add weights to the data
X_y_train = np.concatenate((X_y_train_ttH, X_y_train_ttZ), axis=0)          # Combine ttH and ttZ for training.
X_train = np.concatenate((X_y_train[:,:-10], X_y_train[:,-6:]), axis = 1)   # Seprate X ...
y_train = X_y_train[:,-10:-6]                                               # ... and y.
y_train_masses = np.empty((y_train.shape[0],1))                             # From y we will calculate masses - we use those as labels.

vec = TLorentzVector(0,0,0,0)
for i in range(len(y_train)):              # For each event calculate the Higgs/Z boson mass.
	v = y_train[i]
	vec.SetPxPyPzE(v[0],v[1],v[2],v[3])
	y_train_masses[i] = vec.Mag()

X_y_val_ttH, X_y_val_ttZ = add_weights(X_y_val_ttH, X_y_val_ttZ)    # Repeat for val and test...
X_y_val = np.concatenate((X_y_val_ttH, X_y_val_ttZ), axis=0)
X_val = np.concatenate((X_y_val[:,:-10], X_y_val[:,-6:]), axis = 1)
y_val = X_y_val[:,-10:-6]
y_val_masses = np.empty((y_val.shape[0],1))

vec = TLorentzVector(0,0,0,0)
for i in range(len(y_val)):
	v = y_val[i]
	vec.SetPxPyPzE(v[0],v[1],v[2],v[3])
	y_val_masses[i] = vec.Mag()

X_y_test_ttH, X_y_test_ttZ = add_weights(X_y_test_ttH, X_y_test_ttZ)
X_y_test = np.concatenate((X_y_test_ttH, X_y_test_ttZ), axis=0)
X_test = np.concatenate((X_y_test[:,:-10], X_y_test[:,-6:]), axis = 1)
y_test = X_y_test[:,-10:-6]
y_test_masses = np.empty((y_test.shape[0],1))

vec = TLorentzVector(0,0,0,0)
for i in range(len(y_test)):
	v = y_test[i]
	vec.SetPxPyPzE(v[0],v[1],v[2],v[3])
	y_test_masses[i] = vec.Mag()

num_features = X_train.shape[1]-1   # exclude weights
num_samples = X_train.shape[0]


# %%
# batch size 4096
train_generator = DataGenerator(X_train, y_train_masses, n_features = num_features, batch_size=939, shuffle=True, augmentation = True, ceil = False)
#batch size 128
val_generator = DataGenerator(X_val, y_val_masses, n_features = num_features, batch_size=117, shuffle=False, augmentation = False, ceil = True)
test_generator = DataGenerator(X_test, y_test_masses, n_features = num_features, batch_size=117, shuffle=False, augmentation = False, ceil = True)

# %%
""" Get scaler params... """
# # THIS SHOULD BE COMMENTED OUT DURING A REGULAR RUN
# # AND UNCOMMENTED TO RETRAIN THE SCALER
X_train = np.empty((num_samples,num_features), dtype=float)
y_train_masses = np.empty((num_samples,1), dtype=float)
c = 0

for i in range(len(train_generator)):
	rows = len(train_generator[i][0][:])

	X_train[c:c+rows,:] = train_generator[i][0][:]
	y_train_masses[c:c+rows,:] = train_generator[i][1][:]
	c += rows

X_train = X_train[1:,:]
y_train_masses = y_train_masses[1:,:]

# comment this part out when not training the scaler
# X_scaler = StandardScaler()
# y_scaler = StandardScaler()
# X_scaler = X_scaler.fit(X_train)
# y_scaler = y_scaler.fit(y_train_masses)
#
# f = open("../scaler_params/X_scaler_mass_reco_narrow.csv", "w")
# writer = csv.writer(f)
# writer.writerow(X_scaler.mean_)
# writer.writerow(X_scaler.scale_)
# f.close()
#
# f = open("../scaler_params/y_scaler_higgs_masses_narrow.csv", "w")
# writer = csv.writer(f)
# writer.writerow(y_scaler.mean_)
# writer.writerow(y_scaler.scale_)
# f.close()
# print("end of scaler training")
# # %%
# """ Define NN architecture. """
# def baseline_model(num_features):

# 	i = keras.Input(shape = (num_features,))
# 	dropout_1 = Dropout(0.2)(i)
# 	dense_1 = Dense(10*num_features, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(dropout_1)
# 	relu_1 = Activation(activations.relu)(dense_1)

# 	dropout_2 = Dropout(0.2)(relu_1)
# 	dense_2 = Dense(10*num_features, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(dropout_2)
# 	relu_2 = Activation(activations.relu)(dense_2)

# 	dropout_3 = Dropout(0.2)(relu_2)
# 	dense_3 = Dense(10*num_features, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(dropout_3)
# 	relu_3 = Activation(activations.relu)(dense_3)

# 	dropout_4 = Dropout(0.2)(relu_3)
# 	dense_4 = Dense(10*num_features, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(dropout_4)
# 	relu_4 = Activation(activations.relu)(dense_4)

# 	dropout_5 = Dropout(0.2)(relu_4)
# 	dense_5 = Dense(10*num_features, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(dropout_5)
# 	relu_5 = Activation(activations.relu)(dense_5)

# 	dropout_6 = Dropout(0.2)(relu_5)
# 	dense_6 = Dense(10*num_features, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(dropout_6)
# 	relu_6 = Activation(activations.relu)(dense_6)

# 	dropout_7 = Dropout(0.2)(relu_6)
# 	o = Dense(1, activation='linear', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(dropout_7)

# 	model = keras.Model(i, o)
# 	model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9))
# 	model.summary()
# 	return model

# def scheduler(epoch, lr):
# 	return lr * 0.99




# # tf sequential model (I only pray some of the code I write today will work)
# model = Sequential()
# # first layer
# model.add(Dropout(0.2, input_shape = (num_features,) ))
# model.add(Dense(10*num_features))
# model.add(Activation(activations.relu))

# # second layer
# model.add(Dropout(0.2))
# model.add(Dense(10*num_features))
# model.add(Activation(activations.relu))

# # third layer
# model.add(Dropout(0.2))
# model.add(Dense(10*num_features))
# model.add(Activation(activations.relu))

# # fourth layer
# model.add(Dropout(0.2))
# model.add(Dense(10*num_features))
# model.add(Activation(activations.relu))

# # fifth layer
# model.add(Dropout(0.2))
# model.add(Dense(10*num_features))
# model.add(Activation(activations.relu))

# # sixth layer
# model.add(Dropout(0.2))
# model.add(Dense(10*num_features))
# model.add(Activation(activations.relu))

# # seventh layer
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='linear'))

# model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9))
# model.summary()

# %%
""" Training. Callbacks used are learning rate decay and early stopping. """
# model = baseline_model(num_features)
# callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler)
# history = model.fit(x=train_generator, validation_data = val_generator, epochs=300, verbose=1, callbacks = [callback1, callback2])

# ----------------------------------------------------------------------------------
# BEGINNING OF PYGAD CODE

def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model

    predictions = pygad.kerasga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)

    mse = tensorflow.keras.losses.MeanSquaredError()
    mse_error = mse(data_outputs, predictions).numpy() + 0.00000001
    solution_fitness = 1.0/mse_error

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

#input_layer  = tensorflow.keras.layers.Input(1) # Changed 3 to 1
#dense_layer1 = tensorflow.keras.layers.Dense(5, activation="relu")(input_layer)
#output_layer = tensorflow.keras.layers.Dense(1, activation="linear")(dense_layer1)

input_layer = tensorflow.keras.layers.Input(shape = (num_features,))
dropout_1 = tensorflow.keras.layers.Dropout(0.2)(input_layer)
dense_1 = tensorflow.keras.layers.Dense(10*num_features, kernel_regularizer = l2(0.001), bias_regularizer = l2(0.001))(dropout_1)
relu_1 = tensorflow.keras.layers.Activation(tf.keras.activations.relu)(dense_1)

dropout_2 = tensorflow.keras.layers.Dropout(0.2)(relu_1)
dense_2 = tensorflow.keras.layers.Dense(10*num_features, kernel_regularizer = l2(0.001), bias_regularizer = l2(0.001))(dropout_2)
relu_2 = tensorflow.keras.layers.Activation(tf.keras.activations.relu)(dense_2)

dropout_3 = tensorflow.keras.layers.Dropout(0.2)(relu_2)
dense_3 = tensorflow.keras.layers.Dense(10*num_features, kernel_regularizer = l2(0.001), bias_regularizer = l2(0.001))(dropout_3)
relu_3 = tensorflow.keras.layers.Activation(tf.keras.activations.relu)(dense_3)

dropout_4 = tensorflow.keras.layers.Dropout(0.2)(relu_3)
dense_4 = tensorflow.keras.layers.Dense(10*num_features, kernel_regularizer = l2(0.001), bias_regularizer = l2(0.001))(dropout_4)
relu_4 = tensorflow.keras.layers.Activation(tf.keras.activations.relu)(dense_4)

dropout_5 = tensorflow.keras.layers.Dropout(0.2)(relu_4)
dense_5 = tensorflow.keras.layers.Dense(10*num_features, kernel_regularizer = l2(0.001), bias_regularizer = l2(0.001))(dropout_5)
relu_5 = tensorflow.keras.layers.Activation(tf.keras.activations.relu)(dense_5)

dropout_6 = tensorflow.keras.layers.Dropout(0.2)(relu_5)
dense_6 = tensorflow.keras.layers.Dense(10*num_features, kernel_regularizer = l2(0.001), bias_regularizer = l2(0.001))(dropout_6)
relu_6 = tensorflow.keras.layers.Activation(tf.keras.activations.relu)(dense_6)

dropout_7 = tensorflow.keras.layers.Dropout(0.2)(relu_6)
output_layer = tensorflow.keras.layers.Dense(1, activation=tf.keras.activations.linear, kernel_regularizer=l2(0.001), bias_regularizer = l2(0.001))(dropout_7)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=10)

# Data inputs
#data_inputs = numpy.array([[0.02, 0.1, 0.15],
#                           [0.7, 0.6, 0.8],
#                           [1.5, 1.2, 1.7],
#                           [3.2, 2.9, 3.1]])

data_inputs = X_train

# Data outputs
#data_outputs = numpy.array([[0.1],
#                            [0.6],
#                            [1.3],
#                            [2.5]])
data_outputs = y_train_masses

num_generations = 100 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights # Initial population of network weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation,
                       stop_criteria = "saturate_7",
					   parallel_processing=10)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
# Implement a prediction method here
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Make prediction based on the best solution.
predictions = pygad.kerasga.predict(model=model,
                                    solution=solution,
                                    data=X_test) # Changed from data inputs

print("Predictions : \n", predictions)

mse = tensorflow.keras.losses.MeanSquaredError()
mse_error = mse(y_test_masses, predictions).numpy() # Changed data outputs to y_test
print("Mean Sqaured Error : ", mse_error)  # Changed all instances of MAE to MSE


# ----------------------------------------------------------------------------------



# # Feature importance implementation
# # TAKES A LONG TIME TO RUN, COMMENT OUT UNLESS USING
print("training finished, calculating feature importance...\n")
print("feature names")
print(len(feature_names))
feature_nums = range(len(feature_names))
print(feature_nums)
# myX_train = np.delete(X_train, -1, axis=1) #X_train[:-1] # omit the last character in the array, this should be the weights
# print(X_train.shape)
# print(myX_train.shape)
# print(y_train.shape)
# print(y_train_masses.shape)
# print(myX_train)
# print(y_train_masses)
#
# myX_train = myX_train.astype(np.float64)
# y_train_masses = y_train_masses.astype(np.float64)
# print(np.argwhere(np.isnan(myX_train)))
# print(np.argwhere(np.isnan(y_train_masses)))


# FEATURE IMPORTANCES
# results = permutation_importance(model, X_train, y_train_masses, scoring='neg_mean_squared_error', n_repeats=50)
# importances = results.importances_mean
#
# print(importances.dtype)
# print(importances)
# print(importances.shape)
# print(len(feature_names))
# print(feature_names)
# importances_pd = pd.DataFrame(data={
# 	'Attribute': feature_names,
# 	'Importance': importances
# })
# importances_pd = importances_pd.sort_values(by='Importance', ascending=False)
# print(importances_pd.Importance)
# plt.bar(x=importances_pd['Attribute'], height=importances_pd['Importance'], color='#087E8B')
# # plt.bar([x for x in range(len(importances))], importances)
# plt.title('Feature Importances Obtained by Permutation', size=10)
# plt.xticks(fontsize=5, rotation='vertical')
# # plt.subplots_adjust(bottom=0.15)
# plt.savefig('figures/PA_FeatureImportance.pdf', bbox_inches="tight")
# print(feature_names)

# %%
""" Training history plot. """

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.ylim(0.6,1)
plt.show()
plt.clf()

# %%
""" Save model. """
model.save("models/mass_reco_narrow_btags")

# %%
""" Pyplot histogram. """
#model = keras.models.load_model("models/mass_reco_narrow")

from scipy.optimize import curve_fit
from scipy.stats import crystalball, norm

ttH_eval_generator = DataGenerator(np.concatenate((X_y_test_ttH[:,:-10], X_y_test_ttH[:,-6:]), axis = 1), y_test_masses[:len(X_y_test_ttH)], n_features = num_features, batch_size=64, shuffle=False, augmentation = False, ceil = True)
ttZ_eval_generator = DataGenerator(np.concatenate((X_y_test_ttZ[:,:-10], X_y_test_ttZ[:,-6:]), axis = 1), y_test_masses[len(X_y_test_ttH):len(X_y_test_ttH)+len(X_y_test_ttZ)], n_features = num_features, batch_size=64, shuffle=False, augmentation = False, ceil = True)


# We replace the two lines with comments on them with the values that have been produced by pygad
ttH_y_pred = model.predict(ttH_eval_generator) # THIS IS WHERE WE PREDICT
X_y = ttH_eval_generator.get_all()
ttH_X_eval = X_y[0]
ttH_y_true = X_y[1]

ttZ_y_pred = model.predict(ttZ_eval_generator) # THIS IS WHERE WE PREDICT
X_y = ttZ_eval_generator.get_all()
ttZ_X_eval = X_y[0]
ttZ_y_true = X_y[1]


""" Scale in the opposite direction, to get the mass in MeV. """
with open("../scaler_params/y_scaler_higgs_masses_narrow.csv") as f:
	scaler_params = np.loadtxt(f, delimiter=",")
	y_scaler = scaler_params[0:2]

ttH_y_true = ttH_y_true * y_scaler[1] + y_scaler[0]
ttH_y_pred = ttH_y_pred * y_scaler[1] + y_scaler[0]

ttZ_y_true = ttZ_y_true * y_scaler[1] + y_scaler[0]
ttZ_y_pred = ttZ_y_pred * y_scaler[1] + y_scaler[0]

with open("../scaler_params/X_scaler_mass_reco_narrow.csv") as f:
	scaler_params = np.loadtxt(f, delimiter=",")
	X_scaler = scaler_params[0:2,:]

ttH_X_eval = ttH_X_eval * X_scaler[1] + X_scaler[0]
ttZ_X_eval = ttZ_X_eval * X_scaler[1] + X_scaler[0]

predicted_higgs_masses = []
predicted_Z_masses = []

VIS_TAU_HADR_INDEX = 12
VIS_TAU_LEP_INDEX = 28

plt.figure(figsize=(12,8))

n, bins, patches = plt.hist(ttH_y_pred/1000, bins=np.linspace(0,140,100), alpha = 0.5, density = True, color='blue')
centers = (0.5*(bins[1:]+bins[:-1]))
mean,std = norm.fit(ttH_y_pred/1000)
p0 = [mean, std]
pars1, cov = curve_fit(lambda x, mu, sig : norm.pdf(x, loc=mu, scale=sig), centers, n, p0=p0)
plt.plot(centers, norm.pdf(centers,*pars1), 'r--',linewidth = 2, label='fit before')
plt.annotate("Gaussian fit ttH" "\n" r'$\mu=%.3f$' "\n" r'$\sigma=%.3f$' % (pars1[0], pars1[1]), xy=(0, 1), xytext=(20, -12), va='top',
				 xycoords='axes fraction', textcoords='offset points')

n, bins, patches = plt.hist(ttZ_y_pred/1000, bins=np.linspace(0,140,100), alpha = 0.5, density = True, color='orange')
centers = (0.5*(bins[1:]+bins[:-1]))
mean,std = norm.fit(ttZ_y_pred/1000)
p0 = [mean, std]
pars2, cov = curve_fit(lambda x, mu, sig : norm.pdf(x, loc=mu, scale=sig), centers, n, p0=p0)
plt.plot(centers, norm.pdf(centers,*pars2), 'r--',linewidth = 2, label='fit before')
plt.annotate("Gaussian fit ttZ" "\n" r'$\mu=%.3f$' "\n" r'$\sigma=%.3f$' % (pars2[0], pars2[1]), xy=(0, 1), xytext=(20, -90), va='top',
				 xycoords='axes fraction', textcoords='offset points')

plt.axvline(125,color='blue', linestyle='dashed')
plt.axvline(91.19,color='orange', linestyle='dashed')
plt.title("Mass distribution.\n Single value mass NN output for ttH, ttZ, ttW and tt.")
plt.xlabel("Mass [GeV]")
plt.ylabel("Density")
plt.legend(["125 GeV","91.19 GeV","ttH","ttZ"])
plt.grid()
plt.savefig("../figures/mass_histo_narrow_pyplot.pdf")
plt.show()

print(np.mean(ttH_y_pred/1000), np.std(ttH_y_pred/1000))

# %%
""" Separation curve. """

min_mass = np.min(np.concatenate((ttH_y_pred, ttZ_y_pred)))
max_mass = np.max(np.concatenate((ttH_y_pred, ttZ_y_pred)))


bg_weight = (len(ttZ_y_pred)+len(ttH_y_pred))/2/len(ttZ_y_pred)
h_weight = (len(ttZ_y_pred)+len(ttH_y_pred))/2/len(ttH_y_pred)

n_steps = 1000
step_size = (max_mass-min_mass)/n_steps
current_divider = min_mass

correct_h = np.zeros((n_steps))
correct_bg = np.zeros((n_steps))

incorrect_h = np.zeros((n_steps))
incorrect_bg = np.zeros((n_steps))

steps = np.zeros((n_steps))

for i in range(n_steps):

	current_divider = min_mass+i*step_size
	steps[i] = current_divider

	for h_mass in ttH_y_pred:
		if h_mass < current_divider:
			incorrect_h[i] += 1
		else:
			correct_h[i] += 1

	for z_mass in ttZ_y_pred:
		if z_mass < current_divider:
			correct_bg[i] += 1
		else:
			incorrect_bg[i] += 1

steps = steps/1000
correct_h /= (len(ttH_y_pred)+len(ttZ_y_pred))/100
correct_bg /= (len(ttH_y_pred)+len(ttZ_y_pred))/100

correct_h_area = np.trapz(correct_h, steps)
correct_bg_area = np.trapz(correct_bg, steps)
correct_weighted_area = np.trapz(h_weight*correct_h + bg_weight*correct_bg, steps)


legend_h = "correct Higgs (area: " + str(round(correct_h_area, 2)) + " )"
legend_bg = "correct background (area: " + str(round(correct_bg_area, 2)) + " )"
legend_weighted = "weighted correct total (area: " + str(round(correct_weighted_area, 2)) + " )"

np.save('samples/correct_h_regular4', correct_h)
np.save('samples/correct_bg_regular4', correct_bg)
np.save('samples/correct_weighted_regular4', h_weight*correct_h + bg_weight*correct_bg)
np.save('samples/correct_steps_regular4', steps)

plt.figure(figsize=(12,8))
plt.plot(steps, correct_h, linewidth=3, color='blue')
plt.plot(steps, correct_bg, linewidth=3, color='orange')
plt.plot(steps, h_weight*correct_h + bg_weight*correct_bg, linewidth=3, color='magenta')
plt.title("Mass reconstruction separation curve - NN")
plt.xlabel("separation mass [GeV]")
plt.ylabel("percentage correct [%]")
plt.legend([legend_h, legend_bg, legend_weighted])
plt.grid()
plt.savefig("../figures/mass_histo_narrow_separation_btags.pdf")
plt.show()
# separation S percentage
print("Separation: ")
print(np.max(h_weight*correct_h + bg_weight*correct_bg))

# %%
""" ROOT histogram. """
import atlasplots as aplt
# import root_numpy
from ROOT import TF1, TLine

aplt.set_atlas_style()

H_line = TLine(125.18,0,125.18,1000)
H_line.SetLineStyle(9)
H_line.SetLineWidth(3)
Z_line = TLine(91.19,0,91.19,1000)
Z_line.SetLineStyle(9)
Z_line.SetLineWidth(3)

gauss_fit_1 = TF1("gauss_fit_1","gaus(0)",80,140)
gauss_fit_1.SetParameters(1, 1, 1)

gauss_fit_2 = TF1("gauss_fit_2","gaus(0)",80,140)
gauss_fit_2.SetParameters(1, 1, 1)

ttH = root.TH1F("ttH","",20,80,140)
# root_numpy.fill_hist(ttH, ttH_y_pred.flatten()/1000)
for xeach in (ttH_y_pred.flatten()/1000):
	ttH.Fill(xeach)

# weights are only necessary if you have ttZ information
weights = np.ones(len(ttZ_y_pred))*(len(ttH_y_pred)/len(ttZ_y_pred))

ttZ = root.TH1F("ttZ","",20,80,140)
# root_numpy.fill_hist(ttZ, ttZ_y_pred.flatten()/1000, weights)
for xeach, weight in zip((ttZ_y_pred.flatten()/1000), weights):
	ttZ.Fill(xeach*weight)

fig, ax = aplt.subplots(1, 1)

ttH.Fit("gauss_fit_2","0")
gauss_fit_2.SetNpx(1000)
ax.plot(gauss_fit_2, label="Fit", labelfmt="L", linecolor=root.kRed+1, linewidth=3)

ttZ.Fit("gauss_fit_1","0")
gauss_fit_1.SetNpx(1000)
ax.plot(gauss_fit_1, label=None, labelfmt="L", linecolor=root.kRed+1, linewidth=3)

ttH_fit = ttH.GetFunction("gauss_fit_2")
print(ttH_fit.GetParameter(0))
print(ttH_fit.GetParameter(1))
print(ttH_fit.GetParameter(2))
ax.plot(H_line, linecolor='black', label=None, labelfmt="L")
ax.plot(Z_line, linecolor='black', label=None, labelfmt="L")
ax.plot(ttH, linecolor='blue', linewidth=3, label="ttH", labelfmt="L")
ax.plot(ttZ, linecolor=807, linewidth=3, label="ttZ", labelfmt="L")
ax.set_xlim(80, 140)
ax.set_ylim(0, 380)
ax.set_xlabel("Mass [GeV]")
ax.set_ylabel("Events")
ax.add_margins(top=0.15)
ax.legend(loc=(0.77, 0.7, 0.95, 0.92))
fig.savefig("../figures/mass_histo_narrow_root_btags.pdf")
