
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


# scale_d_b = [100,125,150,175,200]
# scale_u_b = [25,50, 75, 100,150,200]
# #scale stable: only scale when prediction has been stable for x steps
# scale_u_s = [3,4,5,6]
# scale_d_s = [3,4,5,6]
# #scale stable range: define what range is stable
# stable_range = [25, 50, 75, 100]

scale_d_b = np.linspace(50, 500,dtype = int, num=50)
scale_u_b = np.linspace(50, 500,dtype = int, num=50)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out, ywindow):
    X, y = list(), list()

    for i in range(len(sequence)-ywindow-n_steps_in+1):
        # find the end of this pattern
        end_ix = i + n_steps_in

        # gather input and output parts of the pattern
        # print(sequence[end_ix:end_ix+ywindow])
        seq_x, seq_y = sequence[i:end_ix], np.percentile(sequence[end_ix:end_ix+ywindow], 90)
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

def get_best_params(series):
    global params
    global scale_d_b, scale_u_b
    error = np.inf
    best_params = None
    for a in scale_d_b:
        for b in scale_u_b:

            params["scale_down_buffer"] = a
            params["scale_up_buffer"] = b

            rescale_counter = 0
            scaleup = 0
            downscale = 0
            CPU_request = 500
            i = 0
            yrequest_temp = []
            
            while i < len(series):
                if i > 10:

                    p = series[i]
                    if CPU_request - p > params["scale_down_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
                        #print("CPU request wasted")
                        CPU_request = p + params["rescale_buffer"]
                        rescale_counter += 1
                        downscale = 0
                    elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                        
                        #print("CPU request too low")
                        CPU_request = p + params["rescale_buffer"]
                        rescale_counter += 1
                        scaleup = 0
                    scaleup += 1
                    downscale += 1
            
                yrequest_temp.append(CPU_request)
                i += 1 
            # Weighted MSE where CPU usage is higher than request is weighted x times more
            sub = np.subtract(series,yrequest_temp)
            
            for v in range(len(sub)):
                
                if sub[v].flatten() > 0:
                    sub[v] = sub[v] * 10
            MSE = np.square(sub).mean() 
            
            
            if MSE < error: 

                error = MSE
                best_params = params.copy()
                                
                               
    print(error)
    print(best_params)
    return best_params

s_len = 144
params = {
    "window_future": 15, #HW
    "window_past": 1, #HW
    "HW_percentile": 90, #HW
    "season_len": s_len, #HW
    "history_len": 3*s_len, #HW
    "rescale_buffer": 100, # FIX
    "scaleup_count": 18, #FIX
    "scaledown_count": 18, #FIX
    "scale_down_buffer": 100,
    "scale_up_buffer":100
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
    global lstm_model

    
    # 

    # Sine wave
    A = 300
    per = s_len
    B = 2*np.pi/per
    D = 200
    sample = 6*per
    x = np.arange(sample)
    series = A*np.sin(B*x)+D
    #alpha = float(args.alpha)
    series = series * alpha


    # np.random.seed(3)
    noise = np.random.normal(0,int(std),len(series))*(1-alpha)
    series = [sum(x) for x in zip(noise, series)]
    series = [int(i) for i in series]

    series = np.array([1 if i <= 0 else i for i in series]).flatten()

    test_last = s_len*2

    CPU_request = 600
    yrequest = [CPU_request] * test_last
    yhat = []
    Y = []

    i = test_last 

    scaleup = np.inf
    downscale = np.inf
    best = False
    model = None

    rescale_counter = 0

    steps_in, steps_out, n_features, ywindow = 77, 1, 1, 24
    while i <= len(series):

        # What we have seen up until now
        series_part = series[:i]
        
        season = math.ceil((i+1)/s_len)
        history_start_season = season - (params["history_len"]/s_len)
        if history_start_season < 1:
            history_start_season = 1
        history_start = (history_start_season-1) * s_len 
        n = int(i - history_start)
        
        #model = ExponentialSmoothing(series_part[-n:], trend="add", damped=False, seasonal=None)
        #if update model
        raw_seq = series_part
        if i % 144 == 0 or model is None:
            
            model = create_lstm(steps_in, steps_out,n_features, raw_seq, ywindow)


        input_data = np.array(raw_seq[-steps_in:])
        # print(input_data)
        p = lstm_predict(input_data, model,steps_in, n_features)[0]

        # if i % s_len == 0:

        #print(model_fit.params_formatted)
        #p = np.percentile(x, params["HW_percentile"])

        if p < 0:
            p = 0
        Y.append(p)

        # # # # If HW has been running for 1 season
        # if len(Y) >= s_len and len(Y) % s_len == 0:

        #     a = Y[-params["history_len"]:]

        #     a = [x for x in a if not math.isnan(x)]
            
        #     params = get_best_params(a)
        #     best = True

        # if best:
        CPU_request_temp = CPU_request - params["rescale_buffer"]
        if scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
            if CPU_request_temp - p > params["scale_down_buffer"]:
            
                #print("CPU request wasted")
                CPU_request = p + params["rescale_buffer"]
                rescale_counter += 1
                downscale = 0
            elif p - CPU_request_temp > params["scale_up_buffer"]: 
                
                #print("CPU request too low")
                CPU_request = p + params["rescale_buffer"]
                rescale_counter += 1
                scaleup = 0
        scaleup += 1
        downscale += 1

        
        yrequest.append(CPU_request)
        i += 1
    print(rescale_counter)
    
    X = range(len(Y))
    X = [x+test_last for x in X]
    
    



    xrequest = range(len(yrequest))
    #xrequest = [x+test_last for x in xrequest]
    
    
    series_X = range(len(series))
    
    series_X3 = range(len(yhat))

    
    series_X3 = [x+test_last for x in series_X3]
    ax1.plot(X, Y, 'bo-', linewidth=2,label='HW prediction')
    ax1.plot(xrequest,yrequest, 'ro-', linewidth=2,label='CPU requested')
    ax1.plot(series_X, series, 'go-', linewidth=2,label='CPU usage')
    #ax1.plot(series_X3, yhat, 'o-', linewidth=1,label='Lib')

    #Plot estimate of VPA target
    target = np.percentile(series, 90) + 50
    vpa = [target] * len(series_X)
    ax1.plot(series_X, vpa, 'yo-', linewidth=2,label='Estimated VPA target')



    print("---AVERAGE SLACK---")
    skip = params["season_len"]
    #print(yrequest[skip:])
    yrequest.pop()

    reqs = yrequest[skip*3:]
    usages = series[skip*3:]
    vpa_short = vpa[skip*3:]

    # reqs = [350 for i in range(195)]

    avg_slack = np.average(np.subtract(reqs,usages))
    avg_slack_vpa = np.average(np.subtract(vpa[skip*3:],usages))
    
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
    
    # ax1.set_ylim(bottom=-1)
    t = ("CPU autoscaling, alpha: " + str(round(alpha,1)))
    fig.suptitle(t, fontsize=30)
    
    #fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', size=20)
    ax1.tick_params(axis="x", labelsize=20) 
    ax1.tick_params(axis="y", labelsize=20) 
    # ax1.legend(loc='best', fancybox=True, shadow=True, ncol=5, fontsize=24)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=5, fontsize=20)
    ax1.set_xlabel('Observations', fontsize=20)
    ax1.set_ylabel('CPU (millicores)', fontsize=20)



    # Take all (requested CPU - CPU usage)/n

    




    hw_slack = np.subtract(yrequest,series)
    #hw_slack = [0 if i < 0 else i for i in hw_slack]
    vpa_slack = np.subtract(vpa,series)
    #vpa_slack = [0 if i < 0 else i for i in vpa_slack]

    ax2.plot(series_X, hw_slack, 'ro-', linewidth=2, label='Predictive autoscaler')
    ax2.plot(series_X, vpa_slack, 'yo-', linewidth=2, label='Estimated VPA target')

    t2 = "CPU slack, alpha: " + str(round(alpha,1))
    #t2 = ("CPU slack, alpha: " + str(alpha) + "\n VPA avg slack: " + str(int(avg_slack_vpa)) + ", HW avg slack: "+ str(int(avg_slack)))
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

   # manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
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

    
    
