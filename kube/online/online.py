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
# K8s config 
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
lstm_model = None
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


def create_sin_noise(A, D, season_length, total_len):
    # Sine wave 
    B = 2*np.pi/season_length # Period
    x = np.arange(total_len)  # [1, 2, 3, ... total_length]
    series = A*np.sin(B*x)+D  # A is amplitude, D is Y-shift
    alpha = 0.7               # Noise parameter 
    std = 300                 # Noise standard deviation
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
    limits = 1000
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

    except:
        print("get_running_pod excepted")

def get_cpu_usage(api_client):
    # ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods/nginx-deployment-67c998fb9b-gmxqz', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    # response = ret_metrics[0].data.decode('utf-8')
    # a = json.loads(response)
    # return(a["containers"][0]["usage"]["cpu"])
    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/ethowan/pods', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    pod_name = get_running_pod(client, "nginx-deployment", "ethowan")

    ret = None

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
            
            except:
                print("get_cpu_usage excepted, return None")
                return None
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
    except:
        print("get_cpu_requests excepted, return None")
        return None
    
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

    model.fit(X, y, epochs=10, verbose=0)
    
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
    # fit model
    model.fit(X, y, epochs=10, verbose=0)
    
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
    global lstm_model
    #print("cpu usage size:", len(cpu_usages))
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
        if lstm_model is None: 
            lstm_model = create_lstm(steps_in, steps_out,n_features, np.array(cpu_usages), ywindow)
        else:
            lstm_model = update_lstm(steps_in, steps_out, n_features, np.array(cpu_usages[-144*4:]), ywindow, lstm_model)
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
        pred_uppers.append(np.nan)

    pred_x = range(len(pred_targets))
    # pred_x = [i * 15 for i in pred_x] TODO
     
def update_slack_plot():
    global vpa_x, vpa_targets, cpu_x, cpu_usages, vpa_lowers, vpa_uppers, cpu_requests, pred_x, pred_targets, pred_lowers, pred_uppers
    global params
    global cpu_slacks, vpa_slacks
    
    skip = params["season_len"]*2
    if len(cpu_requests)>skip:
        cpu_slacks = np.subtract(cpu_requests[skip:],cpu_usages[skip:])
        vpa_slacks = np.subtract(vpa_targets[skip:],cpu_usages[skip:])


def plot_slack():
    global fig2, ax2
    global cpu_slacks, vpa_slacks

    ax2.clear()
    skip = params["season_len"]*2
    #print(cpu_x[skip:])
    #print(cpu_slacks)
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
    ax1.fill_between(pred_x, pred_lowers, pred_uppers, facecolor='red', alpha=0.3, label="Prediction bounds")  
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
    global vpa_x, vpa_targets, cpu_x, cpu_usages, vpa_lowers, vpa_uppers, cpu_requests, pred_x, pred_targets, pred_lowers, pred_uppers, cooldown
    global plotVPA 
    global data
    global fig1, fig2, ax1, ax2
    plotVPA = True

    # Fast initialize-------------------------------------------------------
    # np.random.seed(13)
    # series = create_sin_noise(A=300, D=200, per=params["season_len"], total_len=2*params["season_len"])
    # cpu_usages = series.tolist()
    cpu_usages = [70.945423, 65.442296, 64.409863, 119.784497, 135.455987, 132.949178, 136.771872, 175.153478, 185.904079, 183.262895, 192.68772, 227.425753, 254.816212, 254.478113, 260.506071, 303.346596, 308.617227, 299.22515, 305.235282, 311.828772, 337.069025, 341.031472, 343.669081, 328.119776, 338.280342, 344.441163, 344.441163, 335.140099, 350.777963, 349.91699, 340.089565, 377.237572, 374.370458, 376.32487, 361.708703, 406.344532, 405.11971, 405.017317, 408.076594, 408.210802, 406.260407, 406.374084, 407.719058, 406.356785, 406.362214, 406.362214, 404.446423, 387.809218, 408.078899, 408.078899, 372.243456, 372.243456, 376.154602, 377.50961, 376.743622, 349.482621, 349.482621, 347.499424, 347.499424, 336.362507, 345.883619, 337.922989, 341.92927, 331.631114, 337.574917, 332.435077, 332.435077, 325.964683, 307.368739, 311.580147, 301.298436, 252.496212, 270.463215, 256.371774, 265.598395, 205.370252, 194.007955, 190.262442, 195.444643, 139.251378, 142.869633, 135.796605, 112.902592, 66.472793, 66.472793, 66.193411, 67.865584, 66.77803, 66.77803, 70.657887, 67.437995, 68.226009, 67.539034, 66.349509, 64.506136, 66.219697, 67.510551, 67.991665, 67.068337, 66.33974, 62.580748, 65.642412, 66.541548, 70.586391, 64.729692, 67.637338, 68.881708, 67.422404, 64.984688, 65.97749, 64.506194, 69.634008, 68.147372, 66.590158, 67.052963, 67.052963, 71.358626, 68.882216, 69.519414, 69.519414, 70.228999, 74.246052, 68.462505, 73.169358, 70.098157, 65.387362, 72.08145, 71.065703, 67.787508, 67.787508, 66.458999, 68.932161, 68.626819, 66.414441, 69.212387, 74.520871, 64.554352, 64.477423, 69.332949, 68.953759, 71.484296, 64.172944, 62.394221, 70.175386, 68.779043, 66.403721, 67.570241, 144.856782, 144.856782, 143.247413, 140.610964, 167.288596, 194.501547, 190.03789, 189.772435, 215.651063, 265.392033, 251.205724, 269.74442, 283.230117, 283.230117, 307.84315, 306.034056, 316.582364, 338.444161, 346.708794, 337.349865, 337.349865, 344.826399, 343.451073, 344.821114, 349.517567, 351.68738, 347.484378, 337.902327, 376.243902, 378.518296, 377.202081, 377.202081, 407.452107, 407.452107, 405.998908, 404.946085, 392.102185, 408.492917, 407.493804, 408.369267, 407.863674, 406.321005, 409.703717, 391.433347, 406.054267, 406.771528, 404.792209, 386.355485, 377.29304, 377.407228, 377.480392, 379.254246, 344.950444, 348.037006, 352.364188, 348.345395, 323.045971, 339.935917, 347.251705, 343.856943, 298.840484, 331.755164, 343.582499, 320.213632, 300.478595, 307.248907, 307.812206, 291.884867, 261.702297, 259.051078, 262.933378, 256.393071, 182.59464, 190.104381, 189.615586, 184.850591, 146.253126, 141.016502, 142.792523, 131.8115, 131.8115, 68.825431, 66.605583, 67.565538, 63.642538, 66.383205, 64.75593, 67.763187, 70.798345, 66.71584, 66.490961, 66.544909, 71.078104, 63.922763, 67.359739, 67.359739, 68.505196, 67.824338, 67.923622, 65.67874, 68.718988, 68.718988, 69.227808, 68.567626, 66.236974, 66.717442, 70.859004, 64.59916, 71.125389, 71.233049, 71.233049, 67.978006, 69.599998, 65.841129, 65.242505, 70.706964, 66.175486, 65.676652, 68.72556, 65.771229, 68.517979, 70.358395, 66.507317, 66.69204, 67.858524, 66.551129, 64.101733, 71.257589, 71.290914, 68.487113, 68.487113, 69.81107, 69.81107, 71.721427, 65.040494, 66.710092, 68.665638, 66.996037, 62.962659, 62.962659, 69.487323, 67.348318, 64.879687, 64.16536]
    
    cpu_usages = cpu_usages[:288]
    cpu_usages = cpu_usages + cpu_usages
    print(len(cpu_usages))
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

    patch(client, 700, 700) 
    #cooldown = params["rescale_cooldown"]

    # ax1.set_xlim(left=params["season_len"]*2) TODO
    # ax2.set_xlim(left=params["season_len"]*2) TODO
    fig1.set_size_inches(15,8)
    fig2.set_size_inches(15,8)

    starttime = time.time()

    while True:
        loopstart = time.time()

        update_main_plot()
        update_slack_plot() 
        
        if data == 'y' or len(pred_targets)%144 == 0:
            print("Saving fig")
            
            
            print(cpu_usages)
            print("--------------------cpu usage size:", len(cpu_usages))
            # Plot figure 
            plot_main()
            plot_slack()
            
            fig1.savefig("./main"+str(len(pred_targets))+".png",bbox_inches='tight')
            fig2.savefig("./slack"+str(len(pred_targets))+".png",bbox_inches='tight')
        
        print("Loop time:", time.time()-loopstart)
        
        sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        print("Sleep time:", sleeptime)
        # sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        time.sleep(sleeptime)
        


    
    
    
if __name__ == '__main__':
    main()
