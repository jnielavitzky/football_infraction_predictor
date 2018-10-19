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
from keras.models import model_from_json
from random import randint

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


# Encode all dataset again to ensure same encoding. Propably better to save it in training.
label_encoder = LabelEncoder()
integer_encoded_home = label_encoder.fit_transform(data[:, [0]])
integer_encoded_away = label_encoder.transform(data[:, [1]])

# To one-hot encoding.
to_categorical(integer_encoded_home).astype(int)
to_categorical(integer_encoded_away).astype(int)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# print(loaded_model.summary())
loaded_model.compile(loss='mae', metrics=['mae'], optimizer="Adam")

# Import data
data = pd.read_csv('./teamstatistics.csv')

# Import data
data_r = pd.read_csv('./referees.csv')
graph_home = []
graph_away = []
for x in range(0, int(sys.argv[1])):

	# Choosing rand inputs.
	index_home = randint(0, 44)
	index_away = randint(0, 44)
	referee_index = randint(0, 20)

	# Get data of random teams.
	home_team = data.values[index_home - 1]
	away_team = data.values[index_away - 1]
	referee = data_r.values[referee_index - 1]

	# Transform to np arrays.
	home_team = np.array(home_team)
	away_team = np.array(away_team)
	referee = np.array(referee)

	# Get one-hot encoding for each team.
	integer_encoded_home = label_encoder.transform([home_team[0]])
	integer_encoded_away = label_encoder.transform([away_team[0]])
	categorical_team_home = to_categorical(integer_encoded_home, 44).astype(int)
	categorical_team_away = to_categorical(integer_encoded_away, 44).astype(int)

	# Get stat data.
	stats = [home_team[4], away_team[4], home_team[3], away_team[3], referee[1]]

	# Horizontaly stack arrays.
	X = np.hstack((categorical_team_home, categorical_team_away))
	X = np.hstack((X, [stats]))

	# predict and save for graph data.
	pred = loaded_model.predict(X)[0]
	graph_home.append(pred[0])
	graph_away.append(pred[1])



# Graph once finished.
plt.plot(graph_home)
plt.plot(graph_away)
plt.xlabel('Random Match #')
plt.legend(['Home Fouls', 'Away Fouls'], loc='upper left')
plt.show()