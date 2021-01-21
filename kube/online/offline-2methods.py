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
    
    model.fit(X[-144:,:,:], y[-144:], epochs=15, verbose=0)

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
    # fit model
    model.fit(X, y, epochs=15, verbose=0)
    
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



def main():

    global params
    global alpha, std

    np.random.seed(13)
    series = create_sin_noise(A=300, D=200, per=params["season_len"], total_len=6*params["season_len"])

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

    steps_in, steps_out, n_features, ywindow = 48, 3, 1, 24

    # Start autoscaling only after we have gathered 2 seasons of data
    while i <= len(series):
        
        # Series up until now
        series_part = series[:i]
        n = calc_n(i)
        # Update/create HW model 
        if i % 20 == 0 or hw_model is None:
            hw_model = ExponentialSmoothing(series_part[-n:], trend="add", seasonal="add", seasonal_periods=params["season_len"])
            model_fit = hw_model.fit()
        # HW prediction
        hw_window = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])
        hw_target = np.percentile(hw_window, params["HW_target"])
        hw_lower = np.percentile(hw_window, params["HW_lower"])
        hw_upper = np.percentile(hw_window, params["HW_upper"])

        # Update/create LSTM model 
        if i % 144 == 0 or model is None:
            # Implementation 1: Create new model every time 
            model = create_lstm(steps_in, steps_out,n_features, series_part,ywindow)

            # Implementation 2: Create model only once, update it by training on new data 
            # if model is None:
            #     model = create_lstm(steps_in, steps_out,n_features, series_part,ywindow, params["lstm_target"])
            # else:
            #     model = update_lstm(steps_in, steps_out,n_features, series_part,ywindow,model, params["lstm_target"])

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

    ax1.set_xlim(left=params["season_len"]*2)
    ax2.set_xlim(left=params["season_len"]*2)

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
    alphas = np.linspace(0.1, 1,dtype = float, num=10)
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

    # ticks = ['0.1', '0.2']
    ticks = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

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

    
