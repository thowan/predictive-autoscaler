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
    "window_future": 24, 
    "window_past": 1, 
    "lstm_target": 90, 
    "lstm_upper": 98, 
    "lstm_lower": 60, 
    "HW_target": 90, 
    "HW_upper": 98, 
    "HW_lower": 60, 
    "season_len": 144, 
    "history_len": 3*144, 
    "rescale_buffer": 120, 
    "rescale_cooldown": 18, 
}

slack_list_vpa = []
slack_list_hw = []
slack_list_lstm = []

perc_list_vpa = []
perc_list_hw = []
perc_list_lstm = []

above_list_vpa = []
above_list_hw = []
above_list_lstm = []

import random

def main():

    global params
    global alpha, std
    
    series_train = np.array([])
    for i in range(24):
        
        s1 = create_sin_noise(A=400, D=200, per=params["season_len"], total_len=1*params["season_len"]) 
        s2 = create_sin_noise(A=50, D=200, per=params["season_len"], total_len=1*params["season_len"])
        if random.random() < 0.5:
            series_train = np.concatenate((series_train, s1))
        else:
            series_train = np.concatenate((series_train, s2))
    
    series = np.array([])
    np.random.seed(18)
    for i in range(10):
        
        s1 = create_sin_noise(A=400, D=200, per=params["season_len"], total_len=1*params["season_len"]) 
        s2 = create_sin_noise(A=50, D=200, per=params["season_len"], total_len=1*params["season_len"])
        if np.random.rand() < 0.5:
            series = np.concatenate((series, s1))
        else:
            series = np.concatenate((series, s2))
    
    # series = [0, 0, 61.743225, 65.812928, 64.688258, 61.283377, 61.283377, 67.654982, 68.506236, 68.069168, 63.33223, 67.350238, 67.350238, 59.135644, 59.135644, 65.063693, 65.367389, 64.644802, 64.644802, 63.671429, 67.479587, 60.369221, 63.746355, 68.426908, 68.426908, 67.848495, 68.379601, 65.033326, 66.127736, 66.072899, 64.005658, 65.626443, 65.626443, 66.947988, 64.115421, 65.943284, 68.052725, 66.706593, 64.079707, 66.359751, 66.652653, 70.252121, 62.923355, 67.126152, 63.567186, 63.522037, 65.607079, 65.990171, 62.548382, 61.127716, 62.720609, 64.266515, 65.87071, 65.87071, 65.245995, 57.512539, 59.729175, 64.033217, 63.277707, 63.300236, 65.711968, 63.283551, 64.754742, 64.754742, 68.782502, 63.018085, 63.018085, 63.563587, 71.889895, 64.555834, 62.135457, 61.660707, 62.18573, 62.18573, 67.263784, 71.418236, 137.838296, 124.973905, 144.046814, 188.035742, 186.621325, 198.083011, 247.179403, 244.414532, 290.002148, 290.002148, 295.501502, 340.193935, 337.603942, 337.603942, 337.75654, 342.075036, 340.059003, 357.540656, 333.370868, 378.765409, 384.203911, 376.488329, 417.234252, 419.647498, 399.808957, 420.418237, 417.371609, 415.687412, 395.31032, 409.665596, 414.328398, 367.193016, 380.588229, 355.168267, 351.698839, 331.657607, 338.393024, 338.393024, 323.316553, 323.316553, 335.339155, 303.564811, 304.098715, 250.799748, 250.799748, 252.688684, 228.040479, 195.5729, 181.70091, 135.338763, 135.338763, 107.944042, 107.944042, 66.244164, 61.989939, 66.483219, 65.621958, 64.934725, 66.901524, 66.69103, 64.781212, 67.526626, 67.774357, 65.041874, 70.976161, 69.392448, 67.182445, 68.313569, 65.735736, 69.748069, 69.879586, 68.653724, 63.957029, 63.507327, 68.12431, 66.650802, 65.715767, 60.000721, 67.136805, 63.062296, 64.018353, 62.040963, 62.040963, 63.679135, 62.154706, 65.181957, 64.068167, 62.955775, 63.776876, 64.813811, 67.133909, 63.064697, 63.378305, 64.617083, 65.588123, 63.81287, 67.709429, 65.932648, 69.467776, 68.323161, 66.588076, 64.669751, 63.069118, 67.945729, 67.22985, 66.832906, 66.23627, 65.847554, 65.495303, 65.495303, 66.514129, 67.329368, 65.083671, 61.357294, 62.94477, 62.212584, 68.336662, 67.07261, 64.864739, 64.958125, 69.507114, 66.610447, 68.525375, 65.489864, 68.253549, 65.888995, 70.468058, 75.914542, 68.394364, 69.441889, 65.384635, 67.455503, 67.455503, 68.250114, 65.357909, 65.483581, 65.483581, 71.977791, 72.614541, 72.614541, 67.588842, 63.59081, 68.024119, 63.750568, 120.006415, 141.780724, 157.363302, 194.491844, 191.004811, 214.548369, 258.842478, 247.114835, 312.71089, 301.050892, 290.10112, 329.702207, 324.740036, 327.989362, 341.644953, 345.155229, 339.429658, 349.754034, 343.547861, 381.285517, 370.047963, 370.047963, 404.025968, 406.407655, 395.729498, 410.945677, 408.927901, 388.462345, 411.756913, 413.74045, 379.053994, 377.120611, 374.794915, 342.678069, 350.37731, 351.175823, 330.527249, 339.084112, 314.821851, 345.557808, 335.729661, 306.96319, 294.292478, 279.71473, 263.332449, 263.332449, 252.101558, 197.453348, 193.687384, 151.860015, 136.625863, 138.461229, 68.300933, 65.568534, 59.355894, 66.259219, 70.144654, 69.622521, 66.224944, 66.347775, 67.625393, 71.684717, 69.288331, 67.552996, 66.877703, 65.358259, 64.09007, 65.972771, 46.733962, 46.733962, 59.836314, 67.122512, 67.122512, 69.136134, 69.565368, 66.371765, 68.182048, 66.810451, 64.643932, 60.62675, 68.080732, 66.749273, 64.798237, 64.798237, 64.878494, 63.181709, 65.372129, 64.0131, 65.70622, 69.175132, 66.861682, 64.719033, 67.670241, 64.220562, 68.25315, 67.834423, 68.560337, 69.022853, 69.313214, 65.118357, 67.406357, 63.376912, 65.045172, 64.989495, 63.718965, 63.345824, 65.449804, 64.155342, 62.886053, 66.267959, 65.485001, 65.855195, 69.179225, 66.506739, 66.506739, 65.672874, 65.700215, 62.53296, 68.086563, 66.499444, 62.409591, 62.711345, 62.659603, 62.659603, 60.395247, 66.284063, 65.326346, 67.663318, 66.458327, 67.466311, 70.133281, 65.957265, 66.595222, 67.34767, 63.437278, 63.25499, 67.629983, 66.231469, 67.961705, 68.507882, 67.745954, 67.745954, 67.248327, 62.504716, 94.801841, 137.00814, 145.935319, 190.619658, 194.037242, 233.892233, 233.892233, 254.712385, 261.488186, 298.071202, 282.161501, 329.107831, 317.119975, 312.571976, 341.597658, 340.412251, 341.760491, 355.978433, 354.902114, 363.21097, 381.047892, 371.724801, 414.718742, 411.074446, 399.373427, 409.639409, 410.727897, 393.188453, 410.719083, 415.898426, 392.934029, 377.301244, 379.155277, 379.155277, 349.539475, 335.751355, 335.751355, 340.206766, 344.070271, 315.490784, 315.490784, 292.290156, 304.459899, 302.161018, 255.155852, 255.155852, 203.190348, 187.090546, 193.63851, 146.449029, 133.040927, 133.040927, 63.658729, 65.872208, 65.872208, 63.785919, 65.557305, 66.257614, 66.257614, 64.719509, 69.977905, 63.764268, 72.086588, 72.086588, 79.028644, 72.453438, 72.453438, 71.439419, 65.766206, 67.981376, 69.559621, 66.739438, 72.002492, 64.164944, 64.655924, 64.272103, 64.272103, 62.007494, 62.007494, 66.338223, 71.645633, 71.645633, 63.56426, 64.926313, 66.310069, 64.223916, 70.085048, 68.419701, 71.077381, 71.36494, 64.996268, 70.869812, 72.071281, 70.278834, 66.573504, 64.79114, 67.027187, 62.453225, 62.453225, 64.880695, 62.802072, 65.361954, 63.988112, 68.05674, 68.05674, 67.980978, 69.14486, 67.490425, 71.864219, 63.439282, 64.259946, 66.794305, 65.083107, 63.974744, 64.744144, 66.039328, 63.075114, 64.599929, 69.433168, 67.252682, 69.62072, 68.596196, 65.898732, 65.898732, 64.297847, 66.054371, 67.160351, 66.757853, 66.550352, 65.17163, 65.384049, 71.176669, 68.427492, 68.479803, 67.810531, 67.852539, 69.594607, 69.594607, 70.910444, 68.897539, 70.113685, 70.113685, 62.509112, 65.087484, 133.947896, 133.947896, 132.924342, 165.104443, 188.169863, 214.01004, 257.470853, 240.649592, 308.856074, 309.477951, 290.347634, 322.716367, 333.424267, 333.424267, 345.628135, 340.405617, 342.477336, 352.506857, 345.058327, 383.199048, 384.811998, 384.811998, 417.381313, 417.381313, 378.148455, 411.174043, 409.55963, 409.55963, 390.938918, 407.548575, 392.441869, 382.398889, 382.757602, 335.682254, 335.682254, 355.582738, 349.104937, 339.425138, 311.241604, 325.765666, 336.398238, 336.398238, 277.215851, 297.545856, 254.817334, 254.882559, 244.950759, 187.152664, 182.643752, 135.005404, 135.005404, 137.099465, 93.870301, 67.082988, 65.723279, 64.320887, 69.562654, 74.428905, 65.510198, 65.510198, 62.158134, 65.298171, 67.352314, 65.257533, 64.268172, 64.268172, 64.564241, 62.156164, 62.156164, 66.038028, 64.117921, 64.117921, 65.074717, 68.808897, 61.544367, 61.544367, 63.719097, 66.237215, 67.173829, 61.888583, 66.855439, 69.535783, 71.224133, 67.677779, 65.835607, 67.041732, 63.31931, 62.19923, 62.33675, 63.082979, 67.750766, 67.750766, 65.903316, 65.559874, 66.629098, 66.887576, 63.563459, 67.011674, 61.641178, 65.090452, 67.282378, 67.282378, 62.226595, 71.180984, 63.285314, 63.569488, 62.547156, 67.443055, 62.172521, 62.172521, 68.00451, 66.581996, 66.23648, 66.102292, 67.380522, 67.380522, 68.871685, 68.871685, 64.193322, 66.494357, 60.046868, 60.046868, 64.822877, 64.009062, 67.008353, 65.557842, 65.797809, 65.164547, 61.930419, 68.302495, 68.302495, 62.909795, 70.484541, 64.651806, 64.789164, 69.860725, 65.664353, 66.606772, 67.59384, 68.336629, 69.146114, 64.815284, 67.871863, 69.952916, 121.959888, 136.68046, 139.548716, 197.362649, 197.362649, 182.25515, 254.721091, 246.582766, 246.582766, 295.386888, 295.060452, 327.11073, 327.11073, 316.816171, 349.175202, 328.152328, 352.463196, 349.979274, 342.156759, 382.986456, 383.414558, 375.817267, 409.974599, 411.372933, 410.578232, 414.90799, 418.508568, 398.545694, 411.562525, 411.407412, 411.788265, 364.112456, 364.112456, 339.263475, 351.129407, 323.364911, 337.980744, 333.326596, 333.326596, 335.167825, 330.241032, 280.122562, 287.321545, 297.07834, 279.535228, 254.74456, 254.374602, 179.04024, 188.421524, 189.60678, 189.60678, 134.465954, 105.120722, 64.622324, 64.037213, 60.449515, 67.349291, 66.888078, 66.888078, 65.276643, 65.22202, 66.109513, 63.836765, 64.270034, 65.032251, 71.107685, 71.107685, 63.12264]
    # series = np.array(series)

    # Two patterns
    # series1 = create_sin_noise(A=300, D=200, per=params["season_len"], total_len=4*params["season_len"])
    # series2 = create_sin_noise(A=700, D=400, per=params["season_len"], total_len=3*params["season_len"])
    # series = np.concatenate((series1,series2), axis=0)
    # series = np.concatenate((series,series1), axis=0)

    scaling_start_index = params["season_len"]*2

    hw_CPU_request = 600
    lstm_CPU_request = 600

    lstm_requests = [lstm_CPU_request] * scaling_start_index
    lstm_targets = []
    lstm_uppers = []
    lstm_lowers = []

    hw_requests = [hw_CPU_request] * scaling_start_index
    hw_targets = []
    hw_uppers = []
    hw_lowers = []
    

    i = scaling_start_index 

    lstm_cooldown = 0
    hw_cooldown = 0

    model = None
    hw_model = None

    steps_in, steps_out, n_features, ywindow = 144, 3, 1, 24

    # Start autoscaling only after we have gathered 2 seasons of data
    while i <= len(series):
        if i % 144 == 0:

            print(i)
        # Series up until now
        series_part = series[:i]
        n = calc_n(i)
        # Update/create HW model 
        if i % 1 == 0 or hw_model is None:
        # if hw_model is None:
            hw_model = ExponentialSmoothing(series_part[-n:], trend="add", seasonal="add", seasonal_periods=params["season_len"])
            model_fit = hw_model.fit()
        # HW prediction
        hw_window = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])
        hw_target = np.percentile(hw_window, params["HW_target"])
        hw_lower = np.percentile(hw_window, params["HW_lower"])
        hw_upper = np.percentile(hw_window, params["HW_upper"])

        
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

        if hw_target < 0:
            hw_target = 0
        if hw_lower < 0:
            hw_lower = 0
        if hw_upper < 0:
            hw_upper = 0

        lstm_targets.append(lstm_target)
        lstm_uppers.append(lstm_upper)
        lstm_lowers.append(lstm_lower)

        hw_targets.append(hw_target)
        hw_uppers.append(hw_upper)
        hw_lowers.append(hw_lower)


        # HW scaling 
        hw_CPU_request_unbuffered = hw_CPU_request - params["rescale_buffer"]
        # If no cool-down
        if (hw_cooldown == 0):
            # If request change greater than 50
            if (abs(hw_CPU_request - (hw_target + params["rescale_buffer"])) > 50):
                # If above upper
                if hw_CPU_request_unbuffered > hw_upper:
                    hw_CPU_request = hw_target + params["rescale_buffer"]
                    hw_cooldown = params["rescale_cooldown"]
                # elseIf under lower
                elif hw_CPU_request_unbuffered < hw_lower: 
                    hw_CPU_request = hw_target + params["rescale_buffer"]
                    hw_cooldown = params["rescale_cooldown"]

        # Reduce cooldown 
        if hw_cooldown > 0:
            hw_cooldown -= 1

        hw_requests.append(hw_CPU_request)

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
    
    ax1.plot(X_targets, hw_targets, 'r-', linewidth=2,label='HW target')

    ax1.plot(X_targets, lstm_targets, 'b--', linewidth=2,label='LSTM target')

    ax1.fill_between(X_targets, hw_lowers, hw_uppers, facecolor='red', alpha=0.3, label="HW bounds")
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
    hw_slack = np.subtract(hw_requests[:-1],series)
    lstm_slack = np.subtract(lstm_requests[:-1],series)
    vpa_slack = np.subtract(vpa_target,series)
    #lstm_slack = [0 if i < 0 else i for i in lstm_slack]
    #vpa_slack = [0 if i < 0 else i for i in vpa_slack]

    ax2.plot(series_X, vpa_slack, 'y--', linewidth=1, label='VPA slack')
    ax2.plot(series_X, hw_slack, 'r-', linewidth=2, label='HW slack')
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
    ax3.plot(X_requests,hw_requests, 'r-', linewidth=2,label='HW requested')
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
    hw_requests.pop()

    lstm_reqs = lstm_requests[skip*2:]
    hw_reqs = hw_requests[skip*2:]
    cpu_usages = series[skip*2:]
    vpa_reqs = vpa_target[skip*2:]
    

    
    lstm_slack = np.subtract(lstm_reqs,cpu_usages)
    hw_slack = np.subtract(hw_reqs,cpu_usages)
    vpa_slack = np.subtract(vpa_reqs,cpu_usages)

    slack_list_vpa.append(vpa_slack)
    slack_list_hw.append(hw_slack)
    slack_list_lstm.append(lstm_slack)

    # Average slack print out

    # avg_slack = np.average(np.subtract(lstm_reqs,cpu_usages))
    # avg_slack_vpa = np.average(np.subtract(vpa_target[skip*2:],cpu_usages))
    # print("HW:", avg_slack)
    # print("VPA:", avg_slack_vpa)

    print("---% TIME ABOVE REQUESTED---")
    lstm_count = 0
    lstm_total = 0
    hw_count = 0
    hw_total = 0
    vpa_count = 0
    vpa_total = 0

    cpu_above_lstm = []
    cpu_above_hw = []
    cpu_above_vpa = []

    for i in range(len(cpu_usages)):
        if cpu_usages[i] > lstm_reqs[i]:
            lstm_count += 1
            amount = cpu_usages[i] - lstm_reqs[i]
            lstm_total += cpu_usages[i] - lstm_reqs[i]
            cpu_above_lstm.append(amount)
        if cpu_usages[i] > hw_reqs[i]:
            hw_count += 1
            amount = cpu_usages[i] - hw_reqs[i]
            hw_total += cpu_usages[i] - hw_reqs[i]
            cpu_above_hw.append(amount)
        if cpu_usages[i] > vpa_reqs[i]:
            vpa_count += 1
            amount = cpu_usages[i] - vpa_reqs[i]
            vpa_total += cpu_usages[i] - vpa_reqs[i] 
            cpu_above_vpa.append(amount)

    above_list_hw.append(cpu_above_hw)
    above_list_lstm.append(cpu_above_lstm)
    above_list_vpa.append(cpu_above_vpa)
    
    perc_above_lstm = round(lstm_count/len(cpu_usages), 4)*100
    perc_above_hw = round(hw_count/len(cpu_usages), 4)*100
    perc_above_vpa = round(vpa_count/len(cpu_usages), 4)*100

    
    perc_list_vpa.append(perc_above_vpa)
    perc_list_lstm.append(perc_above_lstm)
    perc_list_hw.append(perc_above_hw)
    
    # print("Usage above requested LSTM: " + str(perc_above_lstm) + "%")
    # print("Usage above requested HW: " + str(perc_above_hw) + "%")
    # print("Usage above vpa_target: " + str(perc_above_vpa) + "%")

    # BOX 1: Slacks
    # print("Slack list vpa: ", slack_list_vpa)
    # print("Slack list hw: ", slack_list_hw)
    # print("Slack list lstm: ", slack_list_lstm)

    # # BOX 2: CPU above values 
    # print("CPU above lstm: ", above_list_lstm)
    # print("CPU above hw: ", above_list_hw)
    # print("CPU above vpa: ", above_list_vpa)

    # HELP plot 1: %observations above
    print("perc above vpa_target: ", perc_list_vpa)
    print("perc above hw: ", perc_list_hw)
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
    

    # if vpa_count == 0:
    #     above_list_vpa.append(0)
    # else:
    #     above_list_vpa.append(vpa_total/vpa_count)
    # if lstm_count == 0:
    #     above_list_hw.append(0)
    # else:
    #     above_list_hw.append(lstm_total/lstm_count)

    # print("avg above: ", above_list_vpa)
    # print("avg above hw: ", above_list_hw)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=2)
    plt.setp(bp['whiskers'], color=color, linewidth=2)
    plt.setp(bp['caps'], color=color, linewidth=2)
    plt.setp(bp['medians'], color=color, linewidth=2)

if __name__ == '__main__':
    
    std = 300
    # alpha = 0.7
    # main()
    # alphas = np.linspace(0.6, 1,dtype = float, num=1)
    alphas = [0.2,0.4,0.6,0.8]
    for i in range(len(alphas)):
        print(alphas[i])
        alpha = alphas[i]
        main()

    # slack_list_vpa = []
    # slack_list_hw = []
    # slack_list_lstm = []

    

    # data_a = [[1,2,5], [5,7,2,2,5], [7,2,5]]
    # data_b = [[6,4,2], [], [2,3,5,1]]
    # data_c = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]

    data_a = slack_list_vpa
    data_b = slack_list_hw
    data_c = slack_list_lstm

    ticks = ['0.2', '0.4', '0.6', '0.8', '0.9']
    # ticks = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(1,1,1)
    fig4.suptitle("Slack", fontsize=23)
    fig4.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    fig4.set_size_inches(15,8)

    bpl = ax4.boxplot(data_a, whis=(5, 95), positions=np.array(range(len(data_a)))*2.0-0.6, sym='+', widths=0.5, whiskerprops = dict(linestyle='--', linewidth=2))
    bpr = ax4.boxplot(data_b, whis=(5, 95), positions=np.array(range(len(data_b)))*2.0+0.0, sym='+', widths=0.5, whiskerprops = dict(linestyle='--', linewidth=2))
    bpw = ax4.boxplot(data_c, whis=(5, 95), positions=np.array(range(len(data_c)))*2.0+0.6, sym='+', widths=0.5, whiskerprops = dict(linestyle='--', linewidth=2))
    set_box_color(bpl, 'green') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, 'red')
    set_box_color(bpw, 'blue')

    # draw temporary red and blue lines and use them to create a legend
    ax4.plot([], c='green', label='VPA')
    ax4.plot([], c='red', label='HW')
    ax4.plot([], c='blue', label='LSTM')

    ax4.tick_params(axis="x", labelsize=20) 
    ax4.tick_params(axis="y", labelsize=20) 

    leg4 = ax4.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg4_lines = leg4.get_lines()
    plt.setp(leg4_lines, linewidth=5)
    
    ax4.set_xlabel('Alpha', fontsize=20)
    ax4.set_ylabel('CPU (millicores)', fontsize=20)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    fig4.savefig('boxcompare.png')

    # START

    # perc_list_vpa = []
    # perc_list_hw = []
    # perc_list_lstm = []

    # above_list_vpa = []
    # above_list_hw = []
    # above_list_lstm = []
    
    # data_a = [[14,24,3,5,615], [5,7,422,2,5], [7,2,5]]
    # data_b = [[6,4,2], [], [2,3,5,1]]
    # data_c = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]

    data_a = above_list_vpa
    data_b = above_list_hw
    data_c = above_list_lstm


    fig5 = plt.figure(5)
    ax5 = fig5.add_subplot(1,1,1)
    fig5.suptitle("Usage above requested", fontsize=23)
    fig5.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    fig5.set_size_inches(15,8)

    bpl = ax5.boxplot(data_a, whis=(5, 95), positions=np.array(range(len(data_a)))*2.0-0.6, sym='+', widths=0.5, whiskerprops = dict(linestyle='--', linewidth=2))
    bpr = ax5.boxplot(data_b, whis=(5, 95), positions=np.array(range(len(data_b)))*2.0+0.0, sym='+', widths=0.5, whiskerprops = dict(linestyle='--', linewidth=2))
    bpw = ax5.boxplot(data_c, whis=(5, 95), positions=np.array(range(len(data_c)))*2.0+0.6, sym='+', widths=0.5, whiskerprops = dict(linestyle='--', linewidth=2))
    set_box_color(bpl, 'green') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, 'red')
    set_box_color(bpw, 'blue')

    # draw temporary red and blue lines and use them to create a legend
    ax5.plot([], c='green', label='VPA')
    ax5.plot([], c='red', label='HW')
    ax5.plot([], c='blue', label='LSTM')

    ax5.tick_params(axis="x", labelsize=20) 
    ax5.tick_params(axis="y", labelsize=20) 

    leg5 = ax5.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)

    leg5_lines = leg5.get_lines()
    plt.setp(leg5_lines, linewidth=5)

    ax5.set_xlabel('Alpha', fontsize=20)
    ax5.set_ylabel('CPU (millicores)', fontsize=20)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    fig5.savefig('boxcompare1.png')
    ax5.clear()

    # perc above vpa_target:  [3.65, 4.17, 4.34, 3.47, 2.26, 1.5599999999999998, 1.04, 0.35000000000000003, 0.0, 0.0]
    # perc above hw:  [2.78, 1.39, 1.7399999999999998, 0.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # perc above lstm:  [1.7399999999999998, 1.22, 0.35000000000000003, 0.35000000000000003, 0.16999999999999998, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # ---% TIME ABOVE REQUESTED---
    # perc above vpa_target:  [7.8100000000000005, 8.16, 7.12, 6.94, 6.6000000000000005, 6.25, 5.21, 3.1199999999999997, 1.39, 0.0]
    # perc above hw:  [9.55, 10.76, 9.719999999999999, 8.33, 6.25, 4.17, 2.9499999999999997, 0.8699999999999999, 0.0, 0.0]
    # perc above lstm:  [7.470000000000001, 7.470000000000001, 6.6000000000000005, 6.08, 3.3000000000000003, 2.78, 2.08, 0.69, 0.0, 0.0]

    
