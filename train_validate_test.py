from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras import metrics
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# hide tensorflow errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# seed for reproducing same results
seed = int(round(time.time() * 1000)) % 1000
np.random.seed(seed)

# Import data
data = pd.read_csv('./gamescomplete.csv')

# Make data a numpy array
data = data.values
data = np.array(data)

# Label encoder takes the teams name and encodes it to a incremental integer index.
# Liverpool = 0, Man City = 1, ect.
label_encoder = LabelEncoder()
integer_encoded_home = label_encoder.fit_transform(data[:, [0]])
integer_encoded_away = label_encoder.transform(data[:, [1]])

# to_categorical transforms a integer index to a one-hot encoding.
# 0 becomes [1, 0, 0, ...], 1 becomes [0, 1, 0, ...], etc.
categorical_team_home = to_categorical(integer_encoded_home).astype(int)
categorical_team_away = to_categorical(integer_encoded_away).astype(int)

# stats are values like FaulsPerGame, YellowsPerGame, RefereeCardsPerGame, etc.
stats = data[:,[10, 13, 14, 15, 16]]

# this stacks the input data horizontaly. 
# [0, 1, 0, ...], [1, 0, 0, ...] and [0.23, 1.2, ...] becomes [0, 1, 0, ..., 1, 0, 0, ..., 0.23, 1.2, ...]
X = np.hstack((categorical_team_home, categorical_team_away))
X = np.hstack((X, stats))

# this is the output, HomeFauls, AwayFauls
Y = data[:,[2, 3]]

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.5, random_state=seed)

# create the model
model = Sequential()

# dropout disallows overfitting.
model.add(Dropout(0.2, input_shape=(X.shape[1],)))
model.add(Dense(128, activation="softplus"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="softplus"))
model.add(Dense(Y.shape[1]))
model.compile(loss='mean_absolute_error', optimizer="Adam")

# fit the model
history = model.fit(X_train, Y_train, validation_split=0.33, epochs = int(sys.argv[1]), verbose = 1, batch_size = 3000)

# evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=2)

print("Loss on testing: ", scores)

if input("Graph? [y/n]: ") == "y":
	plt.plot(history.history['loss'][0:])
	plt.plot(history.history['val_loss'][0:])
	plt.xlabel('epoch')
	plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'], loc='upper left')
	plt.show()

if input("Save? [y/n]: ") == "y":
	print("Saving...")
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("model.h5")

	print("Saved to disk.")