from pprint import pprint
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import math
import argparse
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.models import load_model

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--alpha", help="Noise alpha", default=1)
parser.add_argument("-s", "--std", help="Standard deviation noise", default=100)
args = parser.parse_args()

alpha = float(args.alpha)
std = int(args.std)

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

#style.use('fivethirtyeight')
fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)
ax3 = fig3.add_subplot(1,1,1)

fig1.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out, ywindow):
    X, y = list(), list()

    for i in range(len(sequence)-ywindow-n_steps_in+1):
        # find the end of this pattern
        end_ix = i + n_steps_in

        # gather input and output parts of the pattern
        # print(sequence[end_ix:end_ix+ywindow])
        seq_x, seq_y = sequence[i:end_ix], [np.percentile(sequence[end_ix:end_ix+ywindow], params["lstm_target"]), np.percentile(sequence[end_ix:end_ix+ywindow], params["lstm_lower"]), np.percentile(sequence[end_ix:end_ix+ywindow], params["lstm_upper"])]
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

def lstm_predict(input_data,model,n_steps_in,n_features):
    # demonstrate prediction
    
    x_input = np.array(trans_foward(input_data))
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    return trans_back(yhat)

def train_test_split(data, n_test):
	return data[:n_test+1], data[-n_test:]

def calc_n(i):
    season = math.ceil((i+1)/params["season_len"])
    history_start_season = season - (params["history_len"]/params["season_len"])
    if history_start_season < 1:
        history_start_season = 1
    history_start = (history_start_season-1) * params["season_len"] 
    n = int(i - history_start)
    return n

def update_lstm(n_steps_in, n_steps_out, n_features,raw_seq, ywindow, model):
    global scaler
    raw_seq = np.array(raw_seq)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(raw_seq.reshape(-1, 1))
    
    dataset = trans_foward(raw_seq)
    # split into samples
    X, y = split_sequence(dataset, n_steps_in, n_steps_out, ywindow)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model.fit(X, y, epochs=15, verbose=0)
    
    # model.fit(X[-144:,:,:], y[-144:], epochs=15, verbose=0)

    return model   


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
    
    # Multi-layer model 
    model.add(LSTM(50, return_sequences=True , input_shape=(n_steps_in, n_features)))
    # model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))

    # Single layer model
    # model.add(LSTM(100, input_shape=(n_steps_in, n_features)))

    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    # fit model
    # model.fit(X, y, epochs=15, verbose=1)

    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # fit model
    history = model.fit(X, y, validation_split=0.3, epochs=30, verbose=0, callbacks=[es, mc])
    # load the saved model
    saved_model = load_model('best_model.h5')
    model = saved_model
    
    
    return model


def create_sin_noise(A, D, per, total_len):
    # Sine wave
    
    B = 2*np.pi/per
    x = np.arange(total_len)
    series = A*np.sin(B*x)+D
    #alpha = float(args.alpha)
    series = series * alpha

    noise = np.random.normal(0,int(std),len(series))*(1-alpha)
    series = [sum(x) for x in zip(noise, series)]
    series = [int(i) for i in series]

    series = np.array([1 if i <= 0 else i for i in series]).flatten()
    return series


params = {
    "window_future": 20, 
    "window_past": 1, 
    "lstm_target": 90, 
    "lstm_upper": 98, 
    "lstm_lower": 60, 
    "season_len": 144, 
    "history_len": 3*144, 
    "rescale_buffer": 120, 
    "rescale_cooldown": 20, 
}

slack_list_vpa = []
slack_list_lstm = []

perc_list_vpa = []
perc_list_lstm = []

above_list_vpa = []
above_list_lstm = []

import random

def main():

    global params
    global alpha, std
    
    # series_train = np.array([])
    # for i in range(24):
        
    #     s1 = create_sin_noise(A=400, D=200, per=params["season_len"], total_len=1*params["season_len"]) 
    #     s2 = create_sin_noise(A=50, D=200, per=params["season_len"], total_len=1*params["season_len"])
    #     if random.random() < 0.5:
    #         series_train = np.concatenate((series_train, s1))
    #     else:
    #         series_train = np.concatenate((series_train, s2))
    
    # series = np.array([])
    # np.random.seed(14)
    # for i in range(10):
        
    #     s1 = create_sin_noise(A=400, D=200, per=params["season_len"], total_len=1*params["season_len"]) 
    #     s2 = create_sin_noise(A=50, D=200, per=params["season_len"], total_len=1*params["season_len"])
    #     if np.random.rand() < 0.5:
    #         series = np.concatenate((series, s1))
    #     else:
    #         series = np.concatenate((series, s2))

    #series_train = np.array([210, 210, 210, 270, 270, 210, 150, 240, 240, 210, 150, 120, 120, 180, 90, 60, 90, 90, 60, 90, 60, 90, 90, 120, 90, 120, 90, 90, 150, 150, 270, 390, 330, 420, 390, 360, 390, 630, 420, 420, 300, 390, 390, 450, 480, 390, 540, 480, 540, 510, 450, 480, 450, 390, 420, 420, 510, 540, 360, 600, 360, 450, 450, 360, 480, 480, 510, 420, 630, 480, 510, 450, 450, 510, 420, 420, 540, 480, 450, 390, 420, 540, 390, 420, 480, 390, 360, 300, 300, 300, 270, 270, 240, 210, 210, 180, 150, 120, 150, 180, 180, 120, 120, 150, 120, 90, 120, 90, 60, 90, 60, 60, 60, 60, 60, 120, 90, 60, 60, 90, 210, 90, 120, 150, 210, 180, 180, 270, 240, 210, 270, 300, 270, 300, 270, 240, 300, 510, 390, 300, 360, 360, 330, 390, 450, 420, 390, 480, 420, 420, 540, 450, 540, 480, 420, 480, 510, 390, 390, 420, 450, 360, 480, 360, 420, 450, 510, 420, 390, 420, 420, 510, 420, 420, 420, 390, 390, 420, 420, 360, 330, 330, 330, 390, 330, 360, 330, 360, 360, 330, 450, 300, 330, 420, 450, 390, 360, 510, 390, 540, 360, 450, 420, 420, 360, 240, 270, 240, 330, 270, 180, 210, 210, 180, 150, 210, 120, 90, 150, 120, 150, 150, 120, 90, 90, 120, 90, 90, 90, 90, 60, 60, 60, 60, 60, 60, 90, 60, 60, 60, 60, 60, 60, 60, 90, 60, 90, 150, 120, 120, 180, 150, 150, 120, 180, 240, 270, 330, 420, 450, 510, 270, 390, 360, 420, 360, 420, 390, 510, 480, 360, 330, 450, 420, 390, 360, 480, 330, 420, 390, 450, 420, 330, 390, 360, 360, 390, 420, 480, 360, 450, 360, 300, 390, 360, 330, 360, 330, 450, 420, 270, 450, 480, 390, 450, 420, 420, 360, 390, 330, 420, 360, 330, 300, 390, 330, 330, 330, 360, 300, 420, 450, 330, 390, 300, 480, 420, 510, 540, 450, 330, 360, 360, 510, 450, 270, 300, 300, 270, 240, 300, 210, 240, 90, 210, 240, 90, 120, 150, 90, 120, 120, 90, 90, 90, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 90, 30, 60, 60, 90, 90, 60, 60, 60, 90, 90, 120, 120, 90, 90, 180, 210, 150, 210, 180, 180, 240, 180, 240, 210, 180, 270, 330, 300, 330, 480, 300, 300, 240, 270, 360, 330, 480, 300, 450, 450, 450, 360, 450, 420, 360, 480, 510, 450, 420, 300, 330, 480, 360, 360, 480, 420, 420, 300, 540, 480, 480, 360, 420, 420, 450, 300, 330, 300, 300, 330, 420, 510, 360, 390, 330, 480, 300, 420, 420, 420, 390, 360, 360, 300, 300, 510, 240, 390, 420, 270, 360, 510, 360, 360, 420, 540, 270, 330, 420, 450, 450, 420, 360, 450, 510, 450, 570, 330, 480, 510, 360, 390, 360, 300, 240, 330, 300, 330, 210, 270, 300, 240, 210, 240, 210, 150, 180, 180, 210, 150, 120, 120, 150, 90, 120, 90, 90, 120, 90, 90, 60, 60, 90, 60, 60, 60, 60, 30, 60, 60, 60, 60, 30, 120, 90, 90, 120, 120, 90, 120, 120, 90, 150, 120, 150, 150, 150, 180, 180, 210, 240, 210, 300, 300, 330, 330, 330, 300, 420, 240, 330, 270, 450, 300, 450, 330, 450, 390, 360, 390, 330, 360, 330, 420, 450, 300, 390, 420, 510, 360, 330, 420, 330, 390, 360, 300, 450, 360, 330, 420, 330, 300, 390, 390, 330, 390, 510, 390, 450, 390, 270, 360, 330, 360, 360, 420, 240, 330, 300, 360, 360, 390, 330, 420, 300, 390, 360, 420, 390, 480, 540, 330, 360, 360, 360, 390, 480, 450, 390, 420, 420, 360, 330, 360, 360, 210, 240, 300, 240, 240, 120, 180, 120, 90, 90, 90, 150, 60, 120, 90, 90, 60, 90, 60, 90, 60, 60, 60, 60, 60, 60, 60, 60, 90, 60, 60, 30, 60, 60, 60, 150, 120, 120, 120, 120, 210, 180, 150, 150, 180, 210, 210, 240, 180, 240, 270, 240, 360, 330, 300, 330, 420, 390, 330, 330, 390, 300, 330, 690, 420, 330, 510, 300, 420, 450, 420, 360, 480, 390, 390, 450, 360, 300, 300, 450, 420, 300, 360, 450, 480, 330, 450, 390, 450, 540, 330, 360, 390, 480, 420, 510, 540, 420, 390, 420, 510, 330, 510, 420, 360, 450, 390, 360, 420, 480, 420, 390, 390, 330, 390, 390, 390, 330, 420, 240, 360, 420, 330, 390, 450, 390, 360, 450, 480, 450, 330, 360, 540, 420, 450, 420, 390, 420, 450, 330, 300, 300, 330, 330, 300, 180, 210, 150, 120, 150, 120, 120, 120, 120, 90, 90, 60, 90, 90, 60, 60, 60, 60, 60, 60, 90, 30, 90, 60, 60, 90, 60, 90, 90, 90, 60, 120, 90, 120, 180, 210, 240, 270, 240, 240, 300, 330, 420, 330, 420, 360, 420, 300, 360, 390, 480, 480, 390, 420, 450, 360, 360, 360, 480, 600, 330, 450, 420, 480, 450, 360, 330, 480, 330, 540, 480, 390, 420, 450, 450, 570, 450, 420, 420, 330, 480, 390, 450, 390, 330, 540, 300, 360, 480, 390, 360, 570, 420, 420, 330, 360, 480, 510, 480, 420, 450, 480, 540, 450, 540, 570, 420, 390, 450, 390, 510, 270, 420, 420, 270, 300, 270, 300, 300, 210, 210, 210, 150, 150, 120, 90, 120, 150, 150, 120, 60, 90, 120, 90, 60, 60, 90, 90, 60, 60, 60, 60, 120, 60, 60, 90, 150, 180, 150, 180, 150, 240, 150, 240, 270, 300, 270, 210, 270, 390, 240, 390, 420, 330, 270, 540, 390, 420, 480, 360, 330, 330, 390, 420, 420, 450, 450, 420, 480, 420, 480, 480, 300, 360, 420, 450, 420, 510, 510, 330, 420, 450, 450, 420, 390, 420, 390, 390, 390, 390, 450, 450, 390, 390, 420, 390, 360, 510, 420, 330, 330, 330, 360, 270, 420, 270, 420, 390, 480, 480, 300, 450, 450, 420, 570, 480, 480, 390, 480, 480, 420, 480, 420, 450, 360, 450, 420, 360, 330])
    #series_train = np.array([690.0, 480.0, 420.0, 600.0, 330.0, 240.0, 210.0, 270.0, 240.0, 270.0, 270.0, 240.0, 270.0, 240.0, 240.0, 180.0, 180.0, 180.0, 240.0, 270.0, 180.0, 330.0, 210.0, 210.0, 300.0, 180.0, 330.0, 240.0, 330.0, 330.0, 210.0, 240.0, 360.0, 270.0, 360.0, 390.0, 300.0, 330.0, 510.0, 570.0, 690.0, 900.0, 570.0, 810.0, 780.0, 870.0, 960.0, 1320.0, 1140.0, 1050.0, 900.0, 840.0, 600.0, 660.0, 840.0, 720.0, 840.0, 630.0, 660.0, 630.0, 690.0, 750.0, 810.0, 870.0, 810.0, 1080.0, 900.0, 750.0, 540.0, 750.0, 660.0, 660.0, 630.0, 750.0, 570.0, 510.0, 600.0, 510.0, 840.0, 540.0, 630.0, 570.0, 540.0, 750.0, 690.0, 510.0, 540.0, 690.0, 480.0, 540.0, 690.0, 630.0, 540.0, 660.0, 450.0, 450.0, 360.0, 360.0, 510.0, 390.0, 390.0, 300.0, 420.0, 420.0, 330.0, 360.0, 270.0, 270.0, 270.0, 270.0, 180.0, 300.0, 330.0, 330.0, 180.0, 180.0, 210.0, 240.0, 180.0, 180.0, 270.0, 480.0, 300.0, 240.0, 360.0, 210.0, 330.0, 180.0, 240.0, 210.0, 360.0, 210.0, 330.0, 270.0, 480.0, 540.0, 660.0, 450.0, 690.0, 900.0, 720.0, 720.0, 900.0, 960.0, 1050.0, 960.0, 840.0, 960.0, 960.0, 870.0, 1050.0, 930.0, 780.0, 930.0, 930.0, 870.0, 810.0, 810.0, 870.0, 870.0, 750.0, 810.0, 810.0, 870.0, 1230.0, 750.0, 960.0, 930.0, 1140.0, 780.0, 660.0, 930.0, 930.0, 930.0, 840.0, 870.0, 1020.0, 840.0, 630.0, 900.0, 780.0, 720.0, 960.0, 780.0, 750.0, 630.0, 660.0, 840.0, 540.0, 630.0, 570.0, 510.0, 570.0, 840.0, 810.0, 660.0, 600.0, 810.0, 630.0, 630.0, 630.0, 690.0, 540.0, 480.0, 690.0, 720.0, 690.0, 600.0, 750.0, 690.0, 660.0, 600.0, 570.0, 660.0, 540.0, 600.0, 600.0, 720.0, 480.0, 480.0, 510.0, 480.0, 510.0, 390.0, 390.0, 510.0, 330.0, 360.0, 390.0, 360.0, 540.0, 450.0, 360.0, 480.0, 390.0, 270.0, 330.0, 210.0, 210.0, 300.0, 300.0, 210.0, 270.0, 360.0, 240.0, 210.0, 480.0, 180.0, 270.0, 270.0, 330.0, 300.0, 270.0, 240.0, 300.0, 210.0, 270.0, 390.0, 330.0, 360.0, 330.0, 360.0, 450.0, 360.0, 480.0, 390.0, 540.0, 510.0, 480.0, 540.0, 600.0, 840.0, 840.0, 810.0, 1200.0, 810.0, 870.0, 990.0, 960.0, 900.0, 810.0, 930.0, 840.0, 720.0, 990.0, 750.0, 1140.0, 780.0, 840.0, 1020.0, 870.0, 690.0, 630.0, 870.0, 870.0, 870.0, 1020.0, 780.0, 900.0, 1050.0, 960.0, 900.0, 1020.0, 780.0, 1020.0, 750.0, 750.0, 750.0, 870.0, 630.0, 600.0, 600.0, 630.0, 720.0, 600.0, 450.0, 630.0, 540.0, 720.0, 570.0, 780.0, 690.0, 540.0, 660.0, 840.0, 720.0, 600.0, 480.0, 780.0, 660.0, 840.0, 510.0, 690.0, 390.0, 600.0, 480.0, 450.0, 420.0, 570.0, 510.0, 510.0, 510.0, 540.0, 600.0, 420.0, 480.0, 570.0, 660.0, 420.0, 540.0, 450.0, 570.0, 390.0, 480.0, 540.0, 540.0, 390.0, 300.0, 390.0, 330.0, 390.0, 360.0, 270.0, 300.0, 270.0, 270.0, 270.0, 270.0, 240.0, 240.0, 210.0, 240.0, 270.0, 270.0, 270.0, 270.0, 300.0, 210.0, 240.0, 240.0, 210.0, 300.0, 420.0, 210.0, 330.0, 210.0, 240.0, 300.0, 240.0, 270.0, 300.0, 240.0, 390.0, 360.0, 390.0, 390.0, 690.0, 360.0, 390.0, 660.0, 750.0, 600.0, 630.0, 780.0, 510.0, 660.0, 750.0, 780.0, 780.0, 960.0, 1140.0, 1080.0, 960.0, 960.0, 1200.0, 1110.0, 840.0, 870.0, 870.0, 780.0, 1110.0, 750.0, 900.0, 900.0, 810.0, 750.0, 840.0, 870.0, 810.0, 900.0, 930.0, 840.0, 900.0, 810.0, 870.0, 840.0, 900.0, 810.0, 570.0, 750.0, 570.0, 480.0, 660.0, 720.0, 750.0, 480.0, 540.0, 390.0, 510.0, 750.0, 540.0, 600.0, 660.0, 600.0, 720.0, 660.0, 540.0, 540.0, 750.0, 540.0, 540.0, 690.0, 570.0, 480.0, 510.0, 660.0, 690.0, 360.0, 390.0, 450.0, 480.0, 450.0, 510.0, 480.0, 360.0, 420.0, 420.0, 390.0, 1050.0, 330.0, 300.0, 360.0, 420.0, 360.0, 270.0, 270.0, 270.0, 180.0, 300.0, 240.0, 210.0, 270.0, 240.0, 240.0, 300.0, 330.0, 270.0, 330.0, 180.0, 270.0, 270.0, 210.0, 390.0, 330.0, 240.0, 210.0, 180.0, 210.0, 240.0, 240.0, 270.0, 300.0, 270.0, 360.0, 420.0, 330.0, 330.0, 480.0, 660.0, 630.0, 720.0, 720.0, 840.0, 630.0, 1020.0, 630.0, 810.0, 960.0, 690.0, 990.0, 810.0, 1170.0, 1080.0, 1050.0, 780.0, 690.0, 810.0, 1170.0, 600.0, 570.0, 750.0, 810.0, 750.0, 900.0, 780.0, 750.0, 780.0, 690.0, 630.0, 660.0, 840.0, 810.0, 600.0, 900.0, 990.0, 810.0, 810.0, 840.0, 780.0, 750.0, 810.0, 750.0, 750.0, 690.0, 780.0, 870.0, 720.0, 690.0, 660.0, 690.0, 510.0, 600.0, 720.0, 600.0, 510.0, 600.0, 630.0, 570.0, 630.0, 570.0, 570.0, 690.0, 570.0, 660.0, 570.0, 510.0, 600.0, 660.0, 630.0, 600.0, 600.0, 480.0, 480.0, 690.0, 450.0, 480.0, 540.0, 510.0, 510.0, 510.0, 540.0, 480.0, 600.0, 540.0, 270.0, 630.0, 570.0, 360.0, 450.0, 450.0, 630.0, 720.0, 480.0, 450.0, 660.0, 420.0, 630.0, 450.0, 330.0, 390.0, 270.0, 360.0, 300.0, 450.0, 330.0, 330.0, 300.0, 240.0, 210.0, 330.0, 390.0, 420.0, 300.0, 270.0, 240.0, 300.0, 210.0, 240.0, 240.0, 240.0, 150.0, 240.0, 270.0, 390.0, 300.0, 180.0, 300.0, 180.0, 270.0, 210.0, 270.0, 300.0, 240.0, 210.0, 270.0, 270.0, 180.0, 240.0, 360.0, 300.0, 330.0, 240.0, 450.0, 270.0, 420.0, 450.0, 450.0, 480.0, 570.0, 750.0, 720.0, 690.0, 630.0, 900.0, 750.0, 990.0, 870.0, 810.0, 900.0, 840.0, 870.0, 840.0, 930.0, 1080.0, 840.0, 1140.0, 870.0, 840.0, 1050.0, 930.0, 840.0, 810.0, 810.0, 720.0, 810.0, 600.0, 810.0, 810.0, 750.0, 840.0, 900.0, 720.0, 840.0, 900.0, 930.0, 780.0, 1050.0, 840.0, 1140.0, 1230.0, 960.0, 930.0, 780.0, 810.0, 810.0, 810.0, 600.0, 780.0, 690.0, 690.0, 540.0, 840.0, 510.0, 600.0, 810.0, 690.0, 540.0, 750.0, 900.0, 690.0, 660.0, 840.0, 570.0, 570.0, 540.0, 840.0, 630.0, 840.0, 660.0, 780.0, 630.0, 780.0, 630.0, 780.0, 570.0, 570.0, 780.0, 690.0, 510.0, 600.0, 510.0, 570.0, 600.0, 570.0, 570.0, 480.0, 510.0, 510.0, 480.0, 480.0, 480.0, 510.0, 420.0, 420.0, 420.0, 300.0, 420.0, 390.0, 420.0, 300.0, 360.0, 330.0, 450.0, 240.0, 330.0, 270.0, 270.0, 360.0, 240.0, 270.0, 300.0, 270.0, 210.0, 210.0, 240.0, 270.0, 180.0, 270.0, 270.0, 270.0, 210.0, 270.0, 270.0, 180.0, 240.0, 360.0, 300.0, 300.0, 210.0, 270.0, 330.0, 420.0, 360.0, 390.0, 330.0, 420.0, 480.0, 480.0, 810.0, 450.0, 630.0, 660.0, 600.0, 660.0, 660.0, 780.0, 720.0, 750.0, 750.0, 900.0, 990.0, 720.0, 690.0, 780.0, 630.0, 1140.0, 840.0, 720.0, 840.0, 810.0, 840.0, 900.0, 750.0, 780.0, 720.0, 900.0, 810.0, 720.0, 900.0, 780.0, 810.0, 960.0, 870.0, 840.0, 900.0, 960.0, 900.0, 990.0, 720.0, 960.0, 810.0, 780.0, 810.0, 780.0, 930.0, 630.0, 870.0, 870.0, 780.0, 960.0, 840.0, 750.0, 810.0, 750.0, 900.0, 720.0, 900.0, 870.0, 870.0, 870.0, 630.0, 900.0, 810.0, 690.0, 750.0, 750.0, 660.0, 720.0, 570.0, 660.0, 720.0, 840.0, 810.0, 690.0, 570.0, 780.0, 600.0, 690.0, 690.0, 660.0, 660.0, 750.0, 750.0, 870.0, 660.0, 720.0, 630.0, 630.0, 630.0, 510.0, 720.0, 570.0, 510.0, 540.0, 540.0, 480.0, 600.0, 540.0, 450.0, 420.0, 420.0, 330.0, 300.0, 300.0, 420.0, 420.0, 330.0, 390.0, 390.0, 270.0, 270.0, 210.0, 330.0, 420.0, 360.0, 210.0, 330.0, 240.0, 240.0, 300.0, 210.0, 180.0, 270.0, 240.0, 240.0, 330.0, 240.0, 210.0, 210.0, 210.0, 210.0, 390.0, 390.0, 300.0, 270.0, 330.0, 570.0, 300.0, 360.0, 540.0, 480.0, 600.0, 510.0, 570.0, 810.0, 780.0, 870.0, 750.0, 840.0, 870.0, 930.0, 840.0, 1020.0, 990.0, 810.0, 930.0, 1110.0, 780.0, 960.0, 870.0, 990.0, 1020.0, 630.0, 660.0, 840.0, 690.0, 840.0, 810.0, 870.0, 1200.0, 810.0, 960.0, 990.0, 960.0, 870.0, 810.0, 870.0, 810.0, 810.0, 810.0, 810.0, 930.0, 990.0, 930.0, 840.0, 990.0, 930.0, 930.0, 1050.0, 840.0, 780.0, 930.0, 810.0, 960.0, 840.0, 720.0, 690.0, 660.0, 990.0, 660.0, 600.0, 570.0, 630.0, 570.0, 720.0, 780.0, 660.0, 600.0, 720.0, 810.0, 780.0, 840.0, 690.0, 690.0, 600.0, 660.0, 630.0, 600.0, 600.0, 450.0, 540.0])
    series_train = [920, 859, 799, 739, 679, 607, 769, 584, 438, 430, 423, 416, 408, 401, 393, 386, 378, 371, 363, 356, 349, 341, 334, 326, 311, 341, 336, 346, 341, 320, 320, 240, 243, 317, 440, 245, 383, 440, 440, 440, 440, 407, 293, 372, 433, 391, 400, 420, 441, 503, 566, 629, 689, 739, 827, 947, 1085, 1000, 1077, 1056, 1060, 1192, 1398, 1560, 1722, 1707, 1639, 1571, 1490, 1387, 1336, 1286, 1235, 1190, 1160, 1129, 974, 901, 1016, 1014, 1081, 851, 928, 1029, 1136, 1296, 1335, 1204, 1047, 994, 986, 977, 969, 961, 953, 945, 936, 928, 920, 912, 904, 895, 887, 879, 871, 863, 854, 846, 838, 830, 822, 813, 805, 797, 789, 781, 772, 764, 756, 748, 740, 731, 723, 885, 880, 858, 968, 791, 691, 925, 736, 825, 965, 838, 696, 857, 778, 864, 799, 746, 878, 787, 695, 604, 600, 548, 480, 480, 480, 652, 520, 520, 477, 430, 450, 560, 447, 448, 360, 360, 360, 360, 360, 360, 360, 360, 306, 249, 309, 369, 440, 240, 240, 244, 320, 240, 240, 373, 382, 321, 460, 405, 349, 293, 385, 280, 305, 281, 394, 380, 386, 441, 495, 550, 605, 680, 880, 670, 749, 828, 906, 1181, 1046, 960, 1148, 1376, 1323, 1240, 1159, 1280, 1256, 1218, 1180, 1270, 1136, 1117, 1240, 1240, 1240, 1240, 1226, 1170, 1099, 1090, 1138, 1137, 1016, 1049, 1080, 1153, 1041, 1264, 1446, 1385, 1160, 1018, 1240, 1132, 1158, 1207, 1259, 1310, 1356, 1244, 1133, 951, 990, 1094, 995, 1272, 1054, 1005, 933, 857, 861, 951, 800, 754, 789, 824, 685, 713, 742, 1053, 1045, 918, 838, 913, 935, 846, 879, 911, 746, 951, 926, 889, 849, 809, 948, 838, 790, 767, 833, 814, 747, 800, 800, 903, 901, 804, 707, 640, 653, 666, 679, 664, 648, 665, 557, 577, 678, 484, 511, 640, 632, 609, 557, 503, 474, 446, 417, 389, 360, 438, 355, 280, 308, 400, 400, 351, 315, 304, 293, 282, 612, 332, 396, 436, 409, 356, 289, 319, 349, 505, 456, 523, 582, 616, 560, 664, 700, 724, 749, 774, 799, 1120, 1143, 1458, 1218, 1113, 1156, 1306, 1285, 1268, 1253, 1238, 1223, 1208, 1158, 1210, 1225, 1044, 1510, 1411, 1313, 1215, 1116, 1045, 1072, 1099, 1345, 1169, 887, 843, 1033, 1160, 1160, 1160, 1225, 1270, 1053, 1129, 1232, 1326, 1214, 1058, 1336, 1221, 1070, 1000, 1000, 1000, 1000, 1021, 1068, 1116, 1141, 915, 817, 800, 824, 686, 819, 733, 773, 813, 853, 893, 933, 778, 924, 812, 749, 983, 896, 748, 1011, 924, 1118, 1022, 925, 829, 732, 802, 520, 754, 628, 751, 680, 680, 698, 753, 717, 692, 592, 683, 694, 562, 683, 680, 441, 513, 381, 373, 387, 395, 365, 360, 360, 360, 360, 360, 360, 360, 360, 360, 320, 314, 307, 300, 293, 286, 280, 315, 360, 360, 380, 317, 320, 297, 349, 538, 284, 330, 387, 337, 367, 402, 484, 505, 520, 701, 730, 516, 711, 904, 820, 999, 825, 861, 1033, 1040, 1059, 1251, 1344, 1420, 1496, 1479, 1372, 1280, 1280, 1465, 1136, 1160, 1116, 1068, 1323, 1036, 1110, 1185, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1174, 1122, 1074, 1048, 1022, 1038, 1131, 1148, 1170, 1214, 1147, 1177, 1093, 1134, 1147, 1162, 1170, 1135, 1100, 1053, 989, 925, 862, 798, 990, 660, 971, 948, 800, 652, 701, 538, 836, 764, 852, 818, 945, 918, 891, 720, 963, 916, 870, 823, 776, 730, 837, 730, 676, 892, 852, 739, 626, 513, 602, 626, 633, 620, 607, 639, 619, 600, 580, 560, 540, 520, 500, 480, 535, 560, 435, 401, 497, 525, 552, 360, 360, 319, 297, 384, 310, 288, 320, 339, 320, 367, 437, 377, 338, 360, 291, 378, 490, 477, 403, 289, 257, 293, 325, 393, 389, 456, 533, 517, 454, 440, 440, 456, 501, 547, 592, 637, 724, 814, 866, 904, 960, 1061, 1228, 852, 1257, 980, 1223, 1198, 1186, 1441, 1295, 1036, 1357, 794, 766, 1026, 1072, 1018, 1196, 1030, 1024, 917, 840, 864, 948, 1115, 1005, 817, 1024, 1254, 1080, 1080, 1116, 1025, 1008, 1067, 1019, 928, 987, 1133, 1009, 955, 949, 943, 937, 931, 924, 915, 894, 901, 875, 814, 752, 691, 810, 810, 836, 761, 828, 765, 760, 780, 813, 846, 880, 913, 791, 870, 842, 830, 800, 800, 640, 640, 640, 901, 807, 712, 618, 655, 693, 673, 683, 774, 829, 786, 669, 600, 600, 740, 714, 653, 757, 597, 542, 488, 483, 458, 389, 397, 463, 531, 424, 389, 338, 281, 536, 549, 417, 364, 320, 338, 283, 308, 305, 218, 357, 480, 402, 341, 250, 300, 288, 399, 310, 297, 284, 271, 349, 421, 450, 420, 381, 476, 584, 600, 600, 600, 729, 823, 900, 977, 941, 903, 1039, 1175, 1182, 1286, 1238, 1191, 1179, 1143, 1190, 1320, 1270, 1446, 1256, 1080, 988, 935, 1004, 1041, 1077, 1113, 1161, 984, 1104, 1200, 1217, 1234, 1089, 1118, 1219, 1319, 1376, 1260, 1143, 1209, 1321, 1434, 1535, 1600, 1456, 1271, 1257, 1244, 1043, 1055, 1067, 1079, 1080, 812, 890, 967, 1029, 920, 1099, 788, 1045, 900, 858, 1188, 900, 843, 760, 721, 839, 957, 1074, 1034, 1000, 974, 844, 902, 960, 1018, 760, 798, 932, 1026, 957, 704, 772, 797, 760, 760, 728, 690, 653, 680, 680, 680, 651, 640, 645, 670, 614, 560, 469, 445, 513, 537, 545, 484, 422, 426, 468, 446, 542, 349, 421, 360, 333, 389, 324, 280, 280, 290, 328, 273, 265, 330, 360, 360, 360, 360, 360, 358, 360, 280, 445, 400, 400, 394, 280, 309, 338, 376, 457, 491, 516, 640, 916, 828, 880, 880, 880, 893, 984, 980, 1260, 998, 1257, 1425, 1275, 1125, 990, 1077, 1080, 1096, 1113, 1164, 1005, 1028, 1080, 1126, 988, 1075, 1188, 1045, 1137, 1159, 1134, 1137, 1180, 1241, 962, 1095, 1228, 1071, 1054, 1047, 1068, 1094, 1214, 1033, 904, 1147, 1158, 1116, 1003, 1024, 1037, 1088, 1138, 1189, 1160, 1160, 1093, 925, 1124, 1175, 1136, 1097, 947, 1000, 895, 904, 933, 848, 850, 961, 1003, 1045, 1087, 1000, 786, 984, 920, 880, 1021, 1126, 840, 840, 840, 840, 935, 880, 824, 769, 725, 684, 720, 720, 720, 699, 686, 772, 686, 624, 579, 446, 422, 405, 400, 400, 443, 528, 560, 507, 520, 398, 354, 294, 509, 422, 360, 298, 331, 405, 320, 320, 351, 373, 318, 272, 245, 294, 358, 320, 280, 280, 280, 288, 342, 396, 450, 504, 520, 520, 520, 400, 477, 631, 468, 620, 640, 713, 892, 1066, 1081, 1100, 1121, 1166, 1196, 1225, 1157, 1228, 1357, 1301, 1191, 1081, 1360, 1110, 1172, 1289, 1077, 864, 850, 862, 874, 998, 1029, 1080, 1113, 1147, 1486, 1119, 1176, 1232, 1286, 1317, 1296, 1243, 1083, 1152, 1118, 1084, 1080, 1080, 1247, 1301, 1200, 1131, 1308, 1240, 1276, 1052, 1106, 1279, 1249, 1218, 1188, 1158, 1128, 966, 949, 939, 928, 901, 890, 801, 815, 770, 924, 992, 1034, 856, 818, 868, 1000, 1060, 1057, 1086, 1114, 920, 869, 877, 842, 806, 800, 800]
    series_train = series_train[144:]
    series = series_train
    series_train = np.concatenate((series_train, series_train, series_train, series_train))

    # Two patterns
    # series1 = create_sin_noise(A=300, D=200, per=params["season_len"], total_len=4*params["season_len"])
    # series2 = create_sin_noise(A=700, D=400, per=params["season_len"], total_len=3*params["season_len"])
    # series = np.concatenate((series1,series2), axis=0)
    # series = np.concatenate((series,series1), axis=0)

    scaling_start_index = params["season_len"]*2

    lstm_CPU_request = 1500

    lstm_requests = [lstm_CPU_request] * scaling_start_index
    lstm_targets = []
    lstm_uppers = []
    lstm_lowers = []

    

    i = scaling_start_index 

    lstm_cooldown = 0

    model = None

    steps_in, steps_out, n_features, ywindow = 144, 3, 1, params["window_future"]

    # Start autoscaling only after we have gathered 2 seasons of data
    while i <= len(series):
        if i % 50 == 0:

            print(i)
        # Series up until now
        series_part = series[:i]


        
        if model is None: 
            model = create_lstm(steps_in, steps_out,n_features, np.array(series_train), ywindow)
            
            
        # LSTM prediction
        input_data = np.array(series_part[-steps_in:])
        output_data = lstm_predict(input_data, model,steps_in, n_features)

        lstm_target = output_data[0] # Target percentile value
        lstm_lower = output_data[1] # Lower bound value 
        lstm_upper = output_data[2] # upper bound value 

        # Keep CPU values above 0 
        if lstm_target < 0:
            lstm_target = 0
        if lstm_lower < 0:
            lstm_lower = 0
        if lstm_upper < 0:
            lstm_upper = 0



        lstm_targets.append(lstm_target)
        lstm_uppers.append(lstm_upper)
        lstm_lowers.append(lstm_lower)



        # LSTM scaling
        lstm_CPU_request_unbuffered = lstm_CPU_request - params["rescale_buffer"]
        
        # If no cool-down
        if (lstm_cooldown == 0):
            # If request change greater than 50
            if (abs(lstm_CPU_request - (lstm_target + params["rescale_buffer"])) > 50):
                # If above upper
                if lstm_CPU_request_unbuffered > lstm_upper:
                    lstm_CPU_request = lstm_target + params["rescale_buffer"]
                    lstm_cooldown = params["rescale_cooldown"]
                # elseIf under lower
                elif lstm_CPU_request_unbuffered < lstm_lower: 
                    lstm_CPU_request = lstm_target + params["rescale_buffer"]
                    lstm_cooldown = params["rescale_cooldown"]

        # Reduce cooldown 
        if lstm_cooldown > 0:
            lstm_cooldown -= 1

        lstm_requests.append(lstm_CPU_request)


        i += 1

    
    X_targets = range(len(lstm_targets))
    X_targets = [x+scaling_start_index for x in X_targets]
    X_requests = range(len(lstm_requests))
    series_X = range(len(series))
    
    # PLOT 1 predictions, order: CPU, VPA, HW, LSTM, HW b, LSTM b
    
    ax1.plot(series_X, series, 'y-', linewidth=1,label='CPU usage')

    #Plot estimate of VPA target
    target = np.percentile(series, 90) + 50
    vpa_target = [target] * len(series_X)

    ax1.plot(series_X, vpa_target, 'g--', linewidth=1,label='VPA target')

    ax1.plot(X_targets, lstm_targets, 'b--', linewidth=2,label='LSTM target')

    ax1.fill_between(X_targets, lstm_lowers, lstm_uppers, facecolor='blue', alpha=0.3, label="LSTM bounds")    




    t = ("CPU prediction, alpha: " + str(round(alpha,1)))
    fig1.suptitle(t, fontsize=23)
    
    ax1.tick_params(axis="x", labelsize=20) 
    ax1.tick_params(axis="y", labelsize=20) 

    leg1 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg1_lines = leg1.get_lines()
    plt.setp(leg1_lines, linewidth=5)

    ax1.set_xlabel('Observations', fontsize=20)
    ax1.set_ylabel('CPU (millicores)', fontsize=20)

    # PLOT 2 Slack
    lstm_slack = np.subtract(lstm_requests[:-1],series)
    vpa_slack = np.subtract(vpa_target,series)
    #lstm_slack = [0 if i < 0 else i for i in lstm_slack]
    #vpa_slack = [0 if i < 0 else i for i in vpa_slack]

    ax2.plot(series_X, vpa_slack, 'y--', linewidth=1, label='VPA slack')
    ax2.plot(series_X, lstm_slack, 'b--', linewidth=2, label='LSTM slack')
    
    

    t2 = "CPU slack, alpha: " + str(round(alpha,1))
    fig2.suptitle(t2, fontsize=23)
    ax2.tick_params(axis="x", labelsize=20) 
    ax2.tick_params(axis="y", labelsize=20) 
    fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

    leg2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg2_lines = leg2.get_lines()
    plt.setp(leg2_lines, linewidth=5)

    ax2.set_xlabel('Observations', fontsize=20)
    ax2.set_ylabel('CPU (millicores)', fontsize=20)
    # ax2.set_ylim(bottom=-100)
    # ax2.set_ylim(top=505)

    # PLOT 3 Requests

    ax3.plot(series_X, series, 'y-', linewidth=1,label='CPU usage')
    ax3.plot(series_X, vpa_target, 'g--', linewidth=1,label='VPA requested')
    ax3.plot(X_requests,lstm_requests, 'b--', linewidth=2,label='LSTM requested')
    
    

    t3 = "CPU autoscaling, alpha: " + str(round(alpha,1))
    fig3.suptitle(t3, fontsize=23)
    ax3.tick_params(axis="x", labelsize=20) 
    ax3.tick_params(axis="y", labelsize=20) 
    fig3.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    leg3 = ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg3_lines = leg3.get_lines()
    plt.setp(leg3_lines, linewidth=5)

    ax3.set_xlabel('Observations', fontsize=20)
    ax3.set_ylabel('CPU (millicores)', fontsize=20)


    # Print out 

    # print("---AVERAGE SLACK---")
    skip = params["season_len"]
    lstm_requests.pop()

    lstm_reqs = lstm_requests[skip*2:]
    cpu_usages = series[skip*2:]
    vpa_reqs = vpa_target[skip*2:]
    

    
    lstm_slack = np.subtract(lstm_reqs,cpu_usages)
    vpa_slack = np.subtract(vpa_reqs,cpu_usages)

    slack_list_vpa.append(vpa_slack)
    slack_list_lstm.append(lstm_slack)

    print("slack vpa: ", np.mean(slack_list_vpa))
    print("slack lstm: ", np.mean(slack_list_lstm))



    # Average slack print out

    print("---% TIME ABOVE REQUESTED---")
    lstm_count = 0
    lstm_total = 0

    vpa_count = 0
    vpa_total = 0

    cpu_above_lstm = []
    cpu_above_vpa = []

    for i in range(len(cpu_usages)):
        if cpu_usages[i] > lstm_reqs[i]:
            lstm_count += 1
            amount = cpu_usages[i] - lstm_reqs[i]
            lstm_total += cpu_usages[i] - lstm_reqs[i]
            cpu_above_lstm.append(amount)
        if cpu_usages[i] > vpa_reqs[i]:
            vpa_count += 1
            amount = cpu_usages[i] - vpa_reqs[i]
            vpa_total += cpu_usages[i] - vpa_reqs[i] 
            cpu_above_vpa.append(amount)

    above_list_lstm.append(cpu_above_lstm)
    above_list_vpa.append(cpu_above_vpa)
    
    perc_above_lstm = round(lstm_count/len(cpu_usages), 4)*100
    perc_above_vpa = round(vpa_count/len(cpu_usages), 4)*100

    
    perc_list_vpa.append(perc_above_vpa)
    perc_list_lstm.append(perc_above_lstm)
    


    # HELP plot 1: %observations above
    print("perc above vpa_target: ", perc_list_vpa)
    print("perc above lstm: ", perc_list_lstm)

    # ax1.set_xlim(left=params["season_len"]*2)
    # ax2.set_xlim(left=params["season_len"]*2)

    fig1.set_size_inches(15,8)
    fig2.set_size_inches(15,8)
    fig3.set_size_inches(15,8)
    # plt.show()

    fig1.savefig("./pred"+str(int(alpha*10))+".png",bbox_inches='tight')
    fig2.savefig("./slack"+str(int(alpha*10))+".png", bbox_inches="tight")  
    fig3.savefig("./scale"+str(int(alpha*10))+".png", bbox_inches="tight")  

    ax1.clear() 
    ax2.clear()
    ax3.clear()
    


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=2)
    plt.setp(bp['whiskers'], color=color, linewidth=2)
    plt.setp(bp['caps'], color=color, linewidth=2)
    plt.setp(bp['medians'], color=color, linewidth=2)

if __name__ == '__main__':
    
    std = 300
    alpha = 0.7
    main()

