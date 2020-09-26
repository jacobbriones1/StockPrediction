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

def reshape_inputs(inputs,K):
    
    inputs = np.array(inputs)
    return np.reshape(inputs,(inputs.shape[0], K ,1))
    
def create_dataset(ticker, startdate, interval, K, show_plot = False):
    df = stock_df(ticker, startdate, interval='1d')
    
    if show_plot == True:
        # Visualize data
        plot_prices(df)
        
    # Create MinMax scaler
    scaler = MinMaxScaler(feature_range = (0, 1))
    
    data_size = len(list(df['Close'].values))
    
    # Create training data and test data
    train = df.iloc[0:round(data_size*0.85)+1, 1:2].values
    test = df.iloc[round(data_size*0.85)+1:, 1:2].values
    train_sc = scaler.fit_transform(train)
    test_sc = scaler.fit_transform(test)
    
    x_train, y_train =[], []
    
    for i in range(K, len(train_sc)):
        x_train.append(train_sc[(i-K):i])
        y_train.append(train_sc[i])

    x_train = reshape_inputs(list(x_train), K)
    y_train = np.array(y_train)
    
    # Repeat for test data
    x_test, y_test =[], []
    
    for i in range(K, len(test_sc)):
        x_test.append(test_sc[(i-K):i])
        y_test.append(test_sc[i])
        
    x_test = reshape_inputs(list(x_test),K,)
    
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)
    
    

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

# returns pretrained model
def load_model(filepath):
    return tf.keras.models.loadmodel(filepath)

# Trains given model, and saves the entire model.
def train(x_train, y_train, model, num_epochs, batch_size):
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train,epochs = num_epochs, batch_size = batch_size)
    model.save('saved_model\my_model') 

# Generates predictions of test data given a trained model
def predict(x_test, model):
    scaler = MinMaxScaler(feature_range = (0,1))
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price
    
