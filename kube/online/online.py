from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

from pprint import pprint
import time
import json
import math
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import threading

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import seaborn
import seaborn as sns

#--------------------------------------------------------------

# Apply the default theme
sns.set_theme()
# K8s config TODO
config.load_kube_config()
api_client = client.ApiClient()
#api_client = None

# Plots 
fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

# Record VPA or not
plotVPA = True

vpa_x = []
vpa_targets = []
vpa_lowers = []
vpa_uppers = []

cpu_x = []
cpu_usages = []
cpu_requests = []

pred_x = []
pred_targets = []
pred_lowers = []
pred_uppers = []

cpu_slacks = []
vpa_slacks = []

cooldown = 0
model = None
hw_model = None
steps_in, steps_out, n_features, ywindow = 48, 3, 1, 24

#PARAMETERS
params = {
    "window_future": 24, 
    "window_past": 1, 
    "lstm_target": 90, 
    "lstm_upper": 98, 
    "lstm_lower": 60, 
    "pred_target": 90, 
    "HW_upper": 98, 
    "HW_lower": 60, 
    "season_len": 144, 
    "history_len": 3*144, 
    "rescale_buffer": 120, 
    "rescale_cooldown": 18, 
}
#-------------------------------------------------------------------


def create_sin_noise(A, D, per, total_len):
    # Sine wave
    
    B = 2*np.pi/per
    x = np.arange(total_len)
    series = A*np.sin(B*x)+D
    alpha = 0.7
    std = 300
    series = series * alpha

    noise = np.random.normal(0,int(std),len(series))*(1-alpha)
    series = [sum(x) for x in zip(noise, series)]
    series = [int(i) for i in series]

    series = np.array([1 if i <= 0 else i for i in series]).flatten()
    return series

def get_vpa_bounds(api_client):
    
    
    try:
        ret_metrics = api_client.call_api('/apis/autoscaling.k8s.io/v1/namespaces/ethowan/verticalpodautoscalers/my-rec-vpa', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
        response = ret_metrics[0].data.decode('utf-8')
        a = json.loads(response)
        containers = a["status"]["recommendation"]["containerRecommendations"]
        container_index = 0
        
        for c in range(len(containers)):
            if "nginx" in containers[c]["containerName"]:
                container_index = c
                break

        vpa_target = a["status"]["recommendation"]["containerRecommendations"][container_index]["target"]["cpu"]
        lowerBound = a["status"]["recommendation"]["containerRecommendations"][container_index]["lowerBound"]["cpu"]
        upperBound = a["status"]["recommendation"]["containerRecommendations"][container_index]["upperBound"]["cpu"]
    except:
        return (None,None,None)
    
    return(vpa_target, lowerBound, upperBound)

def patch(client, requests, limits):
    # Patch'
    limits = requests
    print(requests)
    v1 = client.AppsV1Api()

    #HARDCODED deployment and container names
    dep = {"spec":{"template":{"spec":{"containers":[{"name":"nginx","resources":{"requests":{"cpu":str(int(requests))+"m"},"limits":{"cpu":str(int(limits))+"m"}}}]}}}}
    resp = v1.patch_namespaced_deployment(name='nginx-deployment',  namespace='ethowan', body=dep)
    print("PATCHED request, limits:", str(int(requests))+"m", str(int(limits))+"m")

def get_running_pod(client, name, namespace):
    try:
        api_instance = client.CoreV1Api()
        pod_list = api_instance.list_namespaced_pod(namespace)
        for pod in pod_list.items:
            # HARDCODED deployment name
            if name in pod.metadata.name and 'Running' in pod.status.phase:
                pod_name = pod.metadata.name
                # print("Found: " + pod_name)
                return pod_name

    except ApiException as e:
        print('Found exception in reading the logs', e)

def get_cpu_usage(api_client):
    # ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods/nginx-deployment-67c998fb9b-gmxqz', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    # response = ret_metrics[0].data.decode('utf-8')
    # a = json.loads(response)
    # return(a["containers"][0]["usage"]["cpu"])
    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/ethowan/pods', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    pod_name = get_running_pod(client, "nginx-deployment", "ethowan")

    for i in range(len(a["items"])):
        if pod_name in a["items"][i]["metadata"]["name"]:
            #print(a["items"][i]["containers"][0]["usage"]["cpu"])

            containers = a["items"][i]["containers"]
            container_index = 0
            
            for c in range(len(containers)):
                
                if "nginx" in containers[c]["name"]:
                    container_index = c
                    break
            
            
            try: 
                ret = a["items"][i]["containers"][container_index]["usage"]["cpu"]
            
            except IndexError:
                time.sleep(0.1)
                return get_cpu_usage(api_client)
            return ret

def get_cpu_requests(client):
    try:
        api_instance = client.CoreV1Api()
        pod_list = api_instance.list_namespaced_pod("ethowan")
        for pod in pod_list.items:
            # HARDCODED deployment name
            if "nginx-deployment" in pod.metadata.name:
                pod_name = pod.metadata.name
        api_response = api_instance.read_namespaced_pod(name=pod_name, namespace='ethowan')

        containers = api_response.spec.containers
        container_index = 0
        
        for c in range(len(containers)):
            
            if "nginx" in containers[c].name:
                container_index = c
                break

        return(api_response.spec.containers[container_index].resources.requests["cpu"])
    except ApiException as e:
        print('Found exception in reading the logs', e)
    
def get_cpu_usage_value(cpu_usage):
    if cpu_usage.endswith('m'):
        cpu_usage = int(cpu_usage[:-1])
        cpu_usage = cpu_usage
    elif cpu_usage.endswith('n'):
        cpu_usage = int(cpu_usage[:-1])
        cpu_usage /= 1000000.0
    elif cpu_usage.endswith('u'):
        cpu_usage = int(cpu_usage[:-1])
        cpu_usage /= 1000.0
    else:
        # case cpu == 0
        cpu_usage = int(cpu_usage)
    return cpu_usage

def get_vpa_bound_values(vpa_target, lowerBound, upperBound):
    if vpa_target.endswith('m'):
        vpa_target = vpa_target[:-1]
    if lowerBound.endswith('m'):
        lowerBound = lowerBound[:-1]
    if upperBound.endswith('m') or upperBound.endswith('G'):
        upperBound = upperBound[:-1]

    vpa_target = int(vpa_target)
    lowerBound = int(lowerBound)
    upperBound = int(upperBound)
    return vpa_target, lowerBound, upperBound

def get_cpu_requested_value(cpu_requested):
    if cpu_requested.endswith('m'):
        cpu_requested = cpu_requested[:-1]
    cpu_requested = int(cpu_requested)
    return cpu_requested

def get_input():
    global data
    while True:
        data = input()

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

def train_test_split(data, n_test):
	return data[:n_test+1], data[-n_test:]

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

def predict_lstm(input_data,model,n_steps_in,n_features):
    # demonstrate prediction
    
    x_input = np.array(trans_foward(input_data))
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    output_data = trans_back(yhat)

    lstm_target = output_data[0] # Target percentile value
    lstm_lower = output_data[1] # Lower bound value 
    lstm_upper = output_data[2] # upper bound value 
    if lstm_target < 0:
        lstm_target = 0
    if lstm_lower < 0:
        lstm_lower = 0
    if lstm_upper < 0:
        lstm_upper = 0

    return lstm_target, lstm_lower, lstm_upper

def predict_HW(current_step):
    global params, cpu_usages
    season = math.ceil((current_step+1)/params["season_len"])
    history_start_season = season - (params["history_len"]/params["season_len"])
    if history_start_season < 1:
        history_start_season = 1
    history_start = (history_start_season-1) * params["season_len"] 
    
    n = int(current_step - history_start)
    print("n: ",n)
    model = ExponentialSmoothing(cpu_usages[-n:], trend="add", seasonal="add",seasonal_periods=params["season_len"])
    model_fit = model.fit()
    hw_window = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])
    pred_target = np.percentile(hw_window, params["pred_target"])
    hw_lower = np.percentile(hw_window, params["HW_lower"])
    hw_upper = np.percentile(hw_window, params["HW_upper"])
    if pred_target < 0:
        pred_target = 0
    if hw_lower < 0:
        hw_lower = 0
    if hw_upper < 0:
        hw_upper = 0

    return pred_target, hw_lower, hw_upper

def update_main_plot():

    global api_client
    global vpa_x, vpa_targets, cpu_x, cpu_usages, vpa_lowers, vpa_uppers, cpu_requests, pred_x, pred_targets, pred_lowers, pred_uppers
    global plotVPA 
    global params, cooldown, model, hw_model, steps_in, steps_out, n_features, ywindow

    # Testing ----------------------------------------
    # print ("Hello")
    # return
    
    # VPA array update
    if plotVPA:
        vpa_target, lowerBound, upperBound = get_vpa_bounds(api_client)
        # If getting VPA recommendations
        if vpa_target is not None:

            vpa_target, lowerBound, upperBound = get_vpa_bound_values(vpa_target, lowerBound, upperBound)
            
            # Update VPA arrays
            vpa_targets.append(vpa_target)
            vpa_lowers.append(lowerBound)
            vpa_uppers.append(upperBound)
            
            # vpa_x = [i * 15 for i in vpa_x] #TODO
        else:
            vpa_targets.append(np.nan)
            vpa_lowers.append(np.nan)
            vpa_uppers.append(np.nan)
        vpa_x = range(len(vpa_targets))

    # CPU array update
    cpu_usage = get_cpu_usage(api_client)
    cpu_requested = get_cpu_requests(client)
    
    if cpu_usage is not None:

        # Update cpu arrays
        cpu_usage = get_cpu_usage_value(cpu_usage)
        # When rescaling, CPU usage falls to 0 as new pod starts up TODO need?
        # if cpu_usage <= 0 and len(cpu_usages) > 0:
        #     cpu_usage = cpu_usages[-1]
        cpu_usages.append(cpu_usage)
        # cpu_x = [i * 15 for i in cpu_x] #TODO
    else: 
        print("cpu_usage is None")
        cpu_usages.append(cpu_usages[-1])

    if cpu_requested is not None:

        cpu_requested = get_cpu_requested_value(cpu_requested)
        cpu_requests.append(cpu_requested)
    else: 

        print("cpu_requested is None")
        cpu_requests.append(cpu_requests[-1])

    cpu_x = range(len(cpu_usages))

    # Prediction 
    scaling_start_index = params["season_len"] * 2
    current_step = len(cpu_usages)

    if current_step >= scaling_start_index: 
        # HW Prediction
        # pred_target, pred_lower, pred_upper = predict_HW(current_step)
        # LSTM prediction
        # TODO model is created using all historical usages
        lstm_model = create_lstm(steps_in, steps_out,n_features, np.array(cpu_usages), ywindow)
        input_data = np.array(cpu_usages[-steps_in:])
        pred_target, pred_lower, pred_upper = predict_lstm(input_data, lstm_model,steps_in, n_features)
        
    

        pred_targets.append(pred_target)
        pred_lowers.append(pred_lower)
        pred_uppers.append(pred_upper)
        

        # Scaling 
        cpu_request_unbuffered = cpu_requested - params["rescale_buffer"]
        # If no cool-down
        if (cooldown == 0):
            # If request change greater than 50
            if (abs(cpu_requested - (pred_target + params["rescale_buffer"])) > 50):
                # If above upper
                if cpu_request_unbuffered > pred_upper:
                    patch(client, pred_target + params["rescale_buffer"], pred_target + params["rescale_buffer"])
                    cooldown = params["rescale_cooldown"]
                # elseIf under lower
                elif cpu_request_unbuffered < pred_lower: 
                    patch(client, pred_target + params["rescale_buffer"], pred_target + params["rescale_buffer"])
                    cooldown = params["rescale_cooldown"]

        # Reduce cooldown 
        if cooldown > 0:
            cooldown -= 1


    else:
        pred_targets.append(np.nan)
        pred_lowers.append(np.nan)
        pred_targets.append(np.nan)

    pred_x = range(len(pred_targets))
    # pred_x = [i * 15 for i in pred_x] TODO
     
def update_slack_plot():
    global vpa_x, vpa_targets, cpu_x, cpu_usages, vpa_lowers, vpa_uppers, cpu_requests, pred_x, pred_targets, pred_lowers, pred_uppers
    global params
    global cpu_slacks, vpa_slacks
    
    ax2.clear()
    skip = params["season_len"]*2
    if len(cpu_requests)>skip:
        cpu_slacks.append(np.subtract(cpu_requests[skip:],cpu_usages[skip:]))
        vpa_slacks.append(np.subtract(vpa_targets[skip:],cpu_usages[skip:]))


def plot_slack():
    global fig2, ax2
    global cpu_slacks, vpa_slacks

    ax2.clear()

    ax2.plot(cpu_x[skip:], cpu_slacks, 'b--', linewidth=2, label='CPU slack')
    ax2.plot(cpu_x[skip:], vpa_slacks, 'g-', linewidth=2, label='VPA slack')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    ax2.set_xlabel('Time (s)', fontsize=20)
    ax2.set_ylabel('CPU (millicores)', fontsize=20)
   
        
        
# Plot the main graph, do not show
# VPA target, CPU requests/usage, LSTM bounds
def plot_main():
    global fig1, ax1
    global vpa_x, vpa_targets, cpu_x, cpu_usages, vpa_lowers, vpa_uppers, cpu_requests, pred_x, pred_targets, pred_lowers, pred_uppers

    ax1.clear()
    
    ax1.plot(vpa_x, vpa_targets, 'g--', linewidth=1,label='VPA target')
    ax1.plot(pred_x, pred_targets, 'r-', linewidth=2,label='Prediction target')
    ax1.plot(cpu_x, cpu_requests, 'b--', linewidth=2, label='CPU requested')
    ax1.plot(cpu_x, cpu_usages, 'y-', linewidth=1,label='CPU usage')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    ax1.set_xlabel('Observation', fontsize=20)
    ax1.set_ylabel('CPU (millicores)', fontsize=20)
    
        


def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    global api_client
    global vpa_x, vpa_targets, cpu_x, cpu_usages, vpa_lowers, vpa_uppers, cpu_requests, pred_x, pred_targets, pred_lowers, pred_uppers
    global plotVPA 
    global data
    global fig1, fig2, ax1, ax2
    plotVPA = True

    # Fast initialize-------------------------------------------------------
    np.random.seed(13)
    series = create_sin_noise(A=300, D=200, per=params["season_len"], total_len=2*params["season_len"])
    cpu_usages = series.tolist()
    cpu_requests = [700]*len(cpu_usages)
    cpu_x = range(len(cpu_usages))

    
    pred_targets = [np.nan]*len(cpu_usages)
    pred_lowers = [np.nan]*len(cpu_usages)
    pred_uppers = [np.nan]*len(cpu_usages)
    pred_x = range(len(cpu_usages))

    
    vpa_targets = [np.nan]*len(cpu_usages)
    vpa_lowers = [np.nan]*len(cpu_usages)
    vpa_uppers = [np.nan]*len(cpu_usages)
    vpa_x = range(len(cpu_usages))
    #--------------------------------------------------------------------

    # Keyboard input
    data = [None]
    input_thread = threading.Thread(target=get_input)
    input_thread.start()

# Plot setups ------------------------------------------
    # Set plot title, legend, labels
    fig1.suptitle('nginx pod metrics', fontsize=23)
    fig1.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

    fig2.suptitle('Slack', fontsize=23)
    fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

    # Plot settings
    ax1.tick_params(axis="x", labelsize=20) 
    ax1.tick_params(axis="y", labelsize=20) 

    ax2.tick_params(axis="x", labelsize=20) 
    ax2.tick_params(axis="y", labelsize=20) 
    
# ---------------------------------------------------------------------

    patch(client, 700, 700) # TODO

    # ax1.set_xlim(left=params["season_len"]*2) TODO
    # ax2.set_xlim(left=params["season_len"]*2) TODO
    fig1.set_size_inches(15,8)
    fig2.set_size_inches(15,8)

    starttime = time.time()

    while True:
        
        update_main_plot()
        update_slack_plot() 
        
        if data == 'y' or len(pred_targets)%500 == 0:
            print("Saving fig")
            

            # Plot figure 
            plot_main()
            plot_slack()
            
            fig1.savefig("./main.png",bbox_inches='tight')
            fig2.savefig("./slack.png", bbox_inches="tight")  
            
        sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        # sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        time.sleep(sleeptime)
        


    
    
    
if __name__ == '__main__':
    main()
