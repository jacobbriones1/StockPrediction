# Methods for creating and using LSTM Networks 
#   for predicting stock prices

#  -Jacob Briones

from StockData import (stock_df, plot_prices)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

def reshape_inputs(inputs,K):
    inputs = np.array(inputs)
    return np.reshape(inputs,(inputs.shape[0], K ,1))
    
def create_dataset(ticker, startdate, interval, K, show_plot = False):
    df = stock_df(ticker, startdate, interval='1d')
    
    if show_plot == True:
        # Visualize data
        plot_prices(df)
    
    # Store Close Prices into array
        
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
    dataset_total = pd.concat((train_data, test), axis = 0)
    test = dataset_total[len(dataset_total) - len(test) - K:].values
    test = test.reshape(-1,1)
   
    x_train, y_train =[], []
    
    for i in range(K, len(train_sc)):
        x_train.append(train_sc[(i-K):i])
        y_train.append(train_sc[i])

    x_train = reshape_inputs(list(x_train), K)
    y_train = np.array(y_train)
    
    return (x_train, y_train), test
    
    
#  Create Neural Network Architecture
#   x: training input
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

#  load pre-trained model
def load_model(filepath):
    return tf.keras.models.load_model(filepath)

# Train a given model on specified stock data
def train(x_train, y_train, model, num_epochs, batch_size, filepath = None):
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train,epochs = num_epochs, batch_size = batch_size)
    if filepath != None:
        model.save(filepath) 
   
        
def predict(test, model, plot = False):
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(test)
    x_sc = scaler.transform(test)
    K = model.layers[0].input_shape[1]
    
    x_test = []
    for i in range(K, len(x_sc)):
        x_test.append(x_sc[i-K: i])
    x_test = reshape_inputs(x_test, K)
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    if plot == True:
        test = test.reshape(len(list(test)))
        df= pd.DataFrame({'Prediction':predicted_price.reshape(len(list(predicted_price))),
                  'Actual':list(test)[40:] })
        fig, ax = plt.subplots(figsize=(15, 12), nrows = 1, ncols = 1)
                
        ax.plot(df['Prediction'],color='red',label='Prediction',marker='o', linestyle='--',lw=1.2)
        ax.plot(df['Actual'],color='teal',label='Actual',marker = 'o', linestyle='-',lw=1.2)
        plt.title('Model Predictiion',fontsize=24)
        ax.legend(fontsize=18)
        ax.set_ylabel("Price", fontsize = 18)
        ax.set_xlabel("Time Step", fontsize = 18)
        ax.grid()
        plt.show()
    
    return predicted_price.reshape(len(list(predicted_price)))
