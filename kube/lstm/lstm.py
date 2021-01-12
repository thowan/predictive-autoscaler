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
from matplotlib import pyplot
from scipy import stats
 
# split a univariate sequence into samples
# def split_sequence(sequence, n_steps_in, n_steps_out):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         # find the end of this pattern
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out
#         # check if we are beyond the sequence
#         if out_end_ix > len(sequence):
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#         print(np.array(X), np.array(y))

#     return np.array(X), np.array(y)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out, ywindow):
    X, y = list(), list()

    for i in range(len(sequence)-ywindow-n_steps_in+1):
        # find the end of this pattern
        end_ix = i + n_steps_in

        # gather input and output parts of the pattern
        # print(sequence[end_ix:end_ix+ywindow])
        seq_x, seq_y = sequence[i:end_ix], np.percentile(sequence[end_ix:end_ix+ywindow], 95)
        X.append(seq_x)
        y.append(seq_y)

    # print(np.array(X), np.array(y))
    return np.array(X), np.array(y)

def trans_foward(arr):
    global scaler
    out_arr = scaler.transform(arr.reshape(-1, 1))
    return out_arr.flatten()

def trans_back(arr):
    global scaler
    out_arr = scaler.inverse_transform(arr.flatten().reshape(-1, 1))
    return out_arr.flatten()

def create_lstm(n_steps_in, n_steps_out, n_features,raw_seq, ywindow):
    global scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(raw_seq.reshape(-1, 1))
    #print("First 10 of raw_seq:", raw_seq[:20])
    dataset = trans_foward(raw_seq)
    # split into samples
    X, y = split_sequence(dataset, n_steps_in, n_steps_out, ywindow)
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
    model.fit(X, y, epochs=15, verbose=1)

    # X, X_valid = train_test_split(X, int(len(X)*0.7))
    # y, y_valid = train_test_split(y, int(len(y)*0.7))
    # history = model.fit(X, y, epochs=30, validation_data=(X_valid, y_valid))
    # pyplot.plot(history.history['loss'])
    # pyplot.plot(history.history['val_loss'])
    # pyplot.title('model train vs validation loss')
    # pyplot.ylabel('loss')
    # pyplot.xlabel('epoch')
    # pyplot.legend(['train', 'validation'], loc='upper right')
    # pyplot.show()
    
    return model

def lstm_predict(input_data,model,n_steps_in,n_features):
    # demonstrate prediction
    
    x_input = np.array(trans_foward(input_data))
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    return trans_back(yhat)

def train_test_split(data, n_test):
	return data[:n_test+1], data[-n_test:]

def main():
    steps_in, steps_out, n_features, ywindow = 77, 1, 1, 24

    #start_pred = 60
    #raw_seq = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    
    # Sine wave
    
    A = 300
    per = 144
    B = 2*np.pi/per
    D = 200
    sample = 5*per
    alpha, std = 0.8, 300
    x = np.arange(sample)
    series = A*np.sin(B*x)+D
    series = series * alpha
    
    noise = np.random.normal(0,int(std),len(series))*(1-alpha)
    series = [sum(x) for x in zip(noise, series)]
    raw_seq = np.array([int(i) for i in series])

    lstm_model = create_lstm(steps_in, steps_out,n_features, raw_seq, ywindow)

    #input_data = np.array([60, 70,80,90])
    start_pred = 150
    input_data = np.array(raw_seq[start_pred-steps_in:start_pred])
    yhat = lstm_predict(input_data, lstm_model,steps_in, n_features)
    print("Pred perc:", stats.percentileofscore(raw_seq[start_pred:start_pred+ywindow], yhat))
    print("Predi:", yhat)
    print("Valid 90th perc:", np.percentile(raw_seq[start_pred:start_pred+ywindow],90))
    start_pred = 250
    input_data = np.array(raw_seq[start_pred-steps_in:start_pred])
    yhat = lstm_predict(input_data, lstm_model,steps_in, n_features)
    print("Pred perc:", stats.percentileofscore(raw_seq[start_pred:start_pred+ywindow], yhat))
    print("Predi:", yhat)
    print("Valid 90th perc:", np.percentile(raw_seq[start_pred:start_pred+ywindow],90))
    start_pred = 350
    input_data = np.array(raw_seq[start_pred-steps_in:start_pred])
    yhat = lstm_predict(input_data, lstm_model,steps_in, n_features)
    print("Pred perc:", stats.percentileofscore(raw_seq[start_pred:start_pred+ywindow], yhat))
    print("Predi:", yhat)
    print("Valid 90th perc:", np.percentile(raw_seq[start_pred:start_pred+ywindow],90))
    start_pred = 400
    input_data = np.array(raw_seq[start_pred-steps_in:start_pred])
    yhat = lstm_predict(input_data, lstm_model,steps_in, n_features)
    print("Pred perc:", stats.percentileofscore(raw_seq[start_pred:start_pred+ywindow], yhat))
    print("Predi:", yhat)
    print("Valid 90th perc:", np.percentile(raw_seq[start_pred:start_pred+ywindow],90))
    start_pred = 500
    input_data = np.array(raw_seq[start_pred-steps_in:start_pred])
    yhat = lstm_predict(input_data, lstm_model,steps_in, n_features)
    print("Pred perc:", stats.percentileofscore(raw_seq[start_pred:start_pred+ywindow], yhat))
    print("Predi:", yhat)
    print("Valid 90th perc:", np.percentile(raw_seq[start_pred:start_pred+ywindow],90))
    start_pred = 544
    input_data = np.array(raw_seq[start_pred-steps_in:start_pred])
    yhat = lstm_predict(input_data, lstm_model,steps_in, n_features)
    print("Pred perc:", stats.percentileofscore(raw_seq[start_pred:start_pred+ywindow], yhat))
    print("Predi:", yhat)
    print("Valid 90th perc:", np.percentile(raw_seq[start_pred:start_pred+ywindow],90))

if __name__ == '__main__':
    
    main()