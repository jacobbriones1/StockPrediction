# Methods for creating and using LSTM Networks for stock market data 
#  predictions.
#  -Jacob Briones

from getStockData import (stock_df, plot_prices)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import os

# Helper method for reshaping data
def reshape_inputs(inputs,K):
    
    inputs = np.array(inputs)
    return np.reshape(inputs,(inputs.shape[0], K ,1))
    
# Create a training set and a test set for a company's stock
#  ticker: company stock abbreviation
#  startdate: Date 'Y-M-D' formatted numerically
#  interval: '1y', '6mo', '1d', etc.
#  show_plot: Returns a plot of the company's historical stock prices
#     if value is set to true.
def create_dataset(ticker, startdate, interval, K, show_plot = False):
    df = stock_df(ticker, startdate, interval= interval)
    
    if show_plot == True:
        # Visualize data
        plot_prices(df)
        
    # Create MinMax scaler
    scaler = MinMaxScaler(feature_range = (0, 1))
    
    # Normalize Data using a minmax scaler
    data_size = len(list(df['Close'].values))
    
    # Create training data and test data
    train = df.iloc[0:round(data_size*0.85)+1, 1:2].values
    
    # Keep one variable as a dataframe for formatting test data
    train_data = df.iloc[0:round(data_size*0.85)+1, 1:2]
    
    # Scale training data
    train_sc = scaler.fit_transform(train)
    
    # Format Test data
    test = df.iloc[round(data_size*0.85)+1:,1:2]
    dataset_total = pd.concat((train, test), axis = 0)
    test = dataset_total[len(dataset_total) - len(test) - K:].values
    
    x_train, y_train =[], []
    
    for i in range(K, len(train_sc)):
        x_train.append(train_sc[(i-K):i])
        y_train.append(train_sc[i])

    x_train = reshape_inputs(list(x_train), K)
    y_train = np.array(y_train)
    
    return (x_train, y_train), test
    
    
#  Create tensorflow model
def create_model(x):
    model = Sequential()
    # Add 3 LSTM layers and 3 dropout layers(for preventing overfitting)
    model.add(LSTM(units = 50, 
                   return_sequences = True, 
                   input_shape = (x.shape[1], 1)))
    
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    return model

# Load pre-trained model
def load_model(filepath):
    return tf.keras.models.load_model(filepath)

# Train a given model
def train(x_train, y_train, model, num_epochs, batch_size):
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train,epochs = num_epochs, batch_size = batch_size)
    model.save('saved_model\model') 

# Generate a model's predictions given test data
#
def predict(test, model):
    test = test.reshape(-1,1)
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(test)
    x_sc = scaler.transform(test)
    K = model.layers[0].input_shape[1]
    
    x_test = []
    for i in range(K, len(x_sc)):
        x_test.append(x_sc[i-K: i])
    x_test = reshape_inputs(x_test, K)
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(x_sc)
    return predicted_price
