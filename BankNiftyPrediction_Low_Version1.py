# Recurrent Neural Network

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Training_Set.csv')
training_set = dataset_train.iloc[:, [3,4]].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(100, 2500):
    X_train.append(training_set_scaled[i-100:i, 0:2])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, input_shape = (X_train.shape[1], 2)))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Test_Set.csv')
real_stock_price = dataset_test.iloc[:, 3:4].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train[['Low','Close']], dataset_test[['Low','Close']]), axis = 0, ignore_index=True)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 100:].values
inputs = inputs.reshape(-1,2)
inputs = sc.transform(inputs)
X_test = []
for i in range(100, 137):
    X_test.append(inputs[i-100:i, 0:2])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price_new = np.zeros(shape=(len(predicted_stock_price), 2))
predicted_stock_price_new[:,0] = predicted_stock_price[:,0]
predicted_stock_price = sc.inverse_transform(predicted_stock_price_new)[:,0]


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Bank Nifty Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Bank Nifty Stock Price')
plt.title('Bank Nifty Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bank Nifty Stock Price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

from numpy import savetxt
savetxt('Predicted_Price.csv', predicted_stock_price, fmt='%0.2f')