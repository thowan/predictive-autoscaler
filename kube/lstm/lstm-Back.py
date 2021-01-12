# univariate multi-step vector-output stacked lstm example
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def trans_foward(arr):
    global scaler
    out_arr = scaler.transform(arr.reshape(-1, 1))
    return out_arr.flatten()

def trans_back(arr):
    global scaler
    out_arr = scaler.inverse_transform(arr.flatten().reshape(-1, 1))
    return out_arr.flatten()

def create_lstm(n_steps_in, n_steps_out, n_features,raw_seq):
    global scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(raw_seq.reshape(-1, 1))
    print("First 10 of raw_seq:", raw_seq[:20])
    dataset = trans_foward(raw_seq)
    # split into samples
    X, y = split_sequence(dataset, n_steps_in, n_steps_out)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model

    model = Sequential()
    #model.add(LSTM(100, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(50, return_sequences=True , input_shape=(n_steps_in, n_features)))
    # model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=100, verbose=0)
    return model

def lstm_predict(input_data,model,n_steps_in,n_features):
    # demonstrate prediction
    x_input = np.array(trans_foward(input_data))
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    return trans_back(yhat)

def main():
    steps_in, steps_out, n_features = 10, 5, 1
    # raw_seq = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    # input_data = np.array([60, 70,80,90])
    start_pred = 30
    
    # Sine wave
    A = 300
    per = 144
    B = 2*np.pi/per
    D = 200
    sample = 2*per
    alpha, std = 0.8, 100
    x = np.arange(sample)
    series = A*np.sin(B*x)+D
    series = series * alpha
    
    noise = np.random.normal(0,int(std),len(series))*(1-alpha)
    series = [sum(x) for x in zip(noise, series)]
    raw_seq = np.array([int(i) for i in series])


    input_data = np.array(raw_seq[start_pred-steps_in:start_pred])


    lstm_model = create_lstm(steps_in, steps_out,n_features, raw_seq)
    yhat = lstm_predict(input_data, lstm_model,steps_in, n_features)
    print("Predi:", yhat)
    print("Valid:", raw_seq[start_pred:start_pred+steps_out])
    

if __name__ == '__main__':
    
    main()