
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
fig = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

fig.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out, ywindow, perc):
    X, y = list(), list()

    for i in range(len(sequence)-ywindow-n_steps_in+1):
        # find the end of this pattern
        end_ix = i + n_steps_in

        # gather input and output parts of the pattern
        # print(sequence[end_ix:end_ix+ywindow])
        seq_x, seq_y = sequence[i:end_ix], [np.percentile(sequence[end_ix:end_ix+ywindow], perc), np.percentile(sequence[end_ix:end_ix+ywindow], 60), np.percentile(sequence[end_ix:end_ix+ywindow], 100)]
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

def update_lstm(n_steps_in, n_steps_out, n_features,raw_seq, ywindow, model, perc):
    global scaler
    raw_seq = np.array(raw_seq)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(raw_seq.reshape(-1, 1))
    
    dataset = trans_foward(raw_seq)
    # split into samples
    X, y = split_sequence(dataset, n_steps_in, n_steps_out, ywindow, perc)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    model.fit(X[-144:,:,:], y[-144:], epochs=15, verbose=1)

    return model   

def create_lstm(n_steps_in, n_steps_out, n_features,raw_seq, ywindow, perc):
    global scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(raw_seq.reshape(-1, 1))
    #print("First 10 of raw_seq:", raw_seq[:20])
    dataset = trans_foward(raw_seq)
    # split into samples
    X, y = split_sequence(dataset, n_steps_in, n_steps_out, ywindow, perc)
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
    # fit model
    model.fit(X, y, epochs=20, verbose=1)

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

s_len = 144
params = {
    "window_future": 15, #HW
    "window_past": 1, #HW
    "HW_percentile": 95, #HW
    "season_len": s_len, #HW
    "history_len": 3*s_len, #HW
    "rescale_buffer": 100, # FIX
    "rescale_cooldown": 18, #FIX
}

avgslack_list_vpa = []
avgslack_list_hw = []

timeabove_vpa = []
timeabove_hw = []

avgabove_vpa = []
avgabove_hw = []



def main():

    global s_len
    global params
    global alpha, std

    np.random.seed(13)
    series = create_sin_noise(A=300, D=200, per=s_len, total_len=6*s_len)

    # Two patterns
    # series1 = create_sin_noise(A=300, D=200, per=s_len, total_len=4*s_len)
    # series2 = create_sin_noise(A=700, D=400, per=s_len, total_len=3*s_len)
    # series = np.concatenate((series1,series2), axis=0)
    # series = np.concatenate((series,series1), axis=0)

    scaling_start_index = s_len*2

    hw_CPU_request = 600
    lstm_CPU_request = 600

    lstm_requests = [lstm_CPU_request] * scaling_start_index

    lstm_targets = []

    i = scaling_start_index 

    lstm_cooldown = 0

    model = None

    steps_in, steps_out, n_features, ywindow = 77, 3, 1, 24

    # Start autoscaling only after we have gathered 2 seasons of data
    while i <= len(series):

        # Series up until now
        series_part = series[:i]
        
        #model = ExponentialSmoothing(series_part[-n:], trend="add", damped=False, seasonal=None)
        
        # Update/create model 
        if i % 144 == 0 or model is None:
            # Implementation 1: Create new model every time 
            # model = create_lstm(steps_in, steps_out,n_features, series_part,ywindow, params["HW_percentile"])

            # Implementation 2: Create model only once, update it by training on new data 
            if model is None:
                model = create_lstm(steps_in, steps_out,n_features, series_part,ywindow, params["HW_percentile"])
            else:
                model = update_lstm(steps_in, steps_out,n_features, series_part,ywindow,model, params["HW_percentile"])


        input_data = np.array(series_part[-steps_in:])

        output_data = lstm_predict(input_data, model,steps_in, n_features)

        target = output_data[0] # Target percentile value
        lower = output_data[1] # Lower bound value 
        upper = output_data[2] # upper bound value 

        # Keep CPU values above 0
        if target < 0:
            target = 0
        if lower < 0:
            lower = 0
        if upper < 0:
            upper = 0

        lstm_targets.append(target)

        lstm_CPU_request_unbuffered = lstm_CPU_request - params["rescale_buffer"]
        
        # If no cool-down
        if (lstm_cooldown == 0):
            # If request change greater than 50
            if (abs(lstm_CPU_request - (target + params["rescale_buffer"])) > 50):
                # If above upper
                if lstm_CPU_request_unbuffered > upper:
                    lstm_CPU_request = target + params["rescale_buffer"]
                    lstm_cooldown = params["rescale_cooldown"]
                # elseIf under lower
                elif lstm_CPU_request_unbuffered < lower: 
                    lstm_CPU_request = target + params["rescale_buffer"]
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
    
    ax1.plot(X_targets, lstm_targets, 'b.-', linewidth=2,label='LSTM target')
    ax1.plot(X_requests,lstm_requests, 'r.-', linewidth=2,label='LSTM requested')
    ax1.plot(series_X, series, 'g.-', linewidth=2,label='CPU usage')

    #Plot estimate of VPA target
    target = np.percentile(series, 90) + 50
    vpa = [target] * len(series_X)
    ax1.plot(series_X, vpa, 'y.-', linewidth=2,label='Estimated VPA target')



    print("---AVERAGE SLACK---")
    skip = params["season_len"]
    lstm_requests.pop()

    reqs = lstm_requests[skip*2:]
    usages = series[skip*2:]
    vpa_short = vpa[skip*2:]


    avg_slack = np.average(np.subtract(reqs,usages))
    avg_slack_vpa = np.average(np.subtract(vpa[skip*2:],usages))
    
    print("HW:", avg_slack)
    print("VPA:", avg_slack_vpa)

    print("---% TIME ABOVE REQUESTED---")
    count = 0
    total = 0
    vpa_count = 0
    vpa_total = 0
    for i in range(len(usages)):
        if usages[i] > reqs[i]:
            count += 1
            total += usages[i] - reqs[i]
        if usages[i] > vpa_short[i]:
            vpa_count += 1
            vpa_total += usages[i] - vpa_short[i] 
        
    perc_above_requested = round(count/len(usages), 4)*100
    perc_above_vpa = round(vpa_count/len(usages), 4)*100
    
    print("Usage above requested: " + str(perc_above_requested) + "%")
    print("Exact: " + str(count/len(usages)))
    print("Usage above vpa target: " + str(perc_above_vpa) + "%")
    print("Exact: " + str(vpa_count/len(usages)))

    print("---AVERAGE USAGE ABOVE REQUESTED---")
    if count > 0:
        print("HW: " + str(total/count))
    if vpa_count > 0:
        print("VPA: "+ str(vpa_total/vpa_count))

    t = ("CPU autoscaling, alpha: " + str(round(alpha,1)))
    fig.suptitle(t, fontsize=30)
    
    ax1.tick_params(axis="x", labelsize=20) 
    ax1.tick_params(axis="y", labelsize=20) 

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=5, fontsize=20)
    ax1.set_xlabel('Observations', fontsize=20)
    ax1.set_ylabel('CPU (millicores)', fontsize=20)



    lstm_slack = np.subtract(lstm_requests,series)
    vpa_slack = np.subtract(vpa,series)
    #lstm_slack = [0 if i < 0 else i for i in lstm_slack]
    #vpa_slack = [0 if i < 0 else i for i in vpa_slack]

    ax2.plot(series_X, lstm_slack, 'ro-', linewidth=2, label='LSTM autoscaler')
    ax2.plot(series_X, vpa_slack, 'yo-', linewidth=2, label='Estimated VPA target')

    t2 = "CPU slack, alpha: " + str(round(alpha,1))

    fig2.suptitle(t2, fontsize=30)
    ax2.tick_params(axis="x", labelsize=20) 
    ax2.tick_params(axis="y", labelsize=20) 
    fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    # ax2.legend(loc='best', fancybox=True, shadow=True, ncol=5, fontsize=24)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=5, fontsize=20)
    ax2.set_xlabel('Observations', fontsize=20)
    ax2.set_ylabel('CPU (millicores)', fontsize=20)
    # ax2.set_ylim(bottom=-100)
    # ax2.set_ylim(top=505)

    #plt.show()
    #plt.savefig('fig1.png', figsize=(19.2,10.8), dpi=100)

    ax1.set_xlim(left=s_len*2)
    ax2.set_xlim(left=s_len*2)
    fig.set_size_inches(15,8)
    fig2.set_size_inches(15,8)
    plt.show()

    fig.savefig("./scale"+str(int(alpha*10))+".png",bbox_inches='tight')
    fig2.savefig("./slack"+str(int(alpha*10))+".png", bbox_inches="tight")  

    ax1.clear() 
    ax2.clear() 
    
    avgslack_list_vpa.append(avg_slack_vpa)
    avgslack_list_hw.append(avg_slack)

    print("avg slack vpa: ", avgslack_list_vpa)
    print("avg slack hw: ", avgslack_list_hw)

    timeabove_vpa.append(perc_above_vpa)
    timeabove_hw.append(perc_above_requested)

    print("perc above vpa: ", timeabove_vpa)
    print("perc above hw: ", timeabove_hw)

    if vpa_count == 0:
        avgabove_vpa.append(0)
    else:
        avgabove_vpa.append(vpa_total/vpa_count)
    if count == 0:
        avgabove_hw.append(0)
    else:
        avgabove_hw.append(total/count)

    print("avg above: ", avgabove_vpa)
    print("avg above hw: ", avgabove_hw)

if __name__ == '__main__':
    # alphas = np.linspace(0.1, 1,dtype = float, num=10)
    std = 300
    # for i in range(len(alphas)):
    #     print(alphas[i])
    #     alpha = alphas[i]
    #     main()
    alpha = 0.7
    main()

    
    
