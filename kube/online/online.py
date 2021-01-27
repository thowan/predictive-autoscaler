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

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dep_name", help="Deployment name", default="nginx-deployment")
parser.add_argument("-c", "--cont_name", help="Container name", default="nginx")
parser.add_argument("-v", "--vpa_name", help="VPA name", default="my-rec-vpa")



args = parser.parse_args()

# Plots 
fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

# Record VPA or not
plotVPA = True

# Use LSTM or HW
use_lstm = True
print("use_lstm:", use_lstm)

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
steps_in, steps_out, n_features, ywindow = 48*2, 3, 1, 24

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
    "history_len": 4*144, 
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
        ret_metrics = api_client.call_api('/apis/autoscaling.k8s.io/v1/namespaces/ethowan/verticalpodautoscalers/' + args.vpa_name, 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
        response = ret_metrics[0].data.decode('utf-8')
        a = json.loads(response)
        containers = a["status"]["recommendation"]["containerRecommendations"]
        container_index = 0
        
        for c in range(len(containers)):
            if args.cont_name in containers[c]["containerName"]:
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


    dep = {"spec":{"template":{"spec":{"containers":[{"name":args.cont_name,"resources":{"requests":{"cpu":str(int(requests))+"m"},"limits":{"cpu":str(int(limits))+"m"}}}]}}}}
    resp = v1.patch_namespaced_deployment(name=args.dep_name,  namespace='ethowan', body=dep)
    print("PATCHED request, limits:", str(int(requests))+"m", str(int(limits))+"m")

def get_running_pod(client, name, namespace):
    try:
        api_instance = client.CoreV1Api()
        pod_list = api_instance.list_namespaced_pod(namespace)
        for pod in pod_list.items:
            
            if name in pod.metadata.name and 'Running' in pod.status.phase:
                pod_name = pod.metadata.name
                # print("Found: " + pod_name)
                return pod_name

    except:
        print("get_running_pod excepted")

def get_cpu_usage(api_client):

    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/ethowan/pods', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    pod_name = get_running_pod(client, args.dep_name, "ethowan")

    ret = None

    for i in range(len(a["items"])):
        if pod_name in a["items"][i]["metadata"]["name"]:
            #print(a["items"][i]["containers"][0]["usage"]["cpu"])

            containers = a["items"][i]["containers"]
            container_index = 0
            
            for c in range(len(containers)):
                
                if args.cont_name in containers[c]["name"]:
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
            
            if args.dep_name in pod.metadata.name:
                pod_name = pod.metadata.name
        api_response = api_instance.read_namespaced_pod(name=pod_name, namespace='ethowan')

        containers = api_response.spec.containers
        container_index = 0
        
        for c in range(len(containers)):
            
            if args.cont_name in containers[c].name:
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

        if use_lstm:
            # LSTM prediction
            # TODO model is created using all historical usages
            if lstm_model is None: 
                lstm_model = create_lstm(steps_in, steps_out,n_features, np.array(cpu_usages), ywindow)
            else:
                lstm_model = update_lstm(steps_in, steps_out, n_features, np.array(cpu_usages[-params["history_len"]:]), ywindow, lstm_model)
            input_data = np.array(cpu_usages[-steps_in:])
            pred_target, pred_lower, pred_upper = predict_lstm(input_data, lstm_model,steps_in, n_features)
        else:
            # HW Prediction
            pred_target, pred_lower, pred_upper = predict_HW(current_step)

        
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
    else:
        cpu_slacks = np.array([])
        vpa_slacks = np.array([])

def plot_slack():
    global fig2, ax2
    global cpu_slacks, vpa_slacks

    ax2.clear()
    skip = params["season_len"]*2
    #print(cpu_x[skip:])
    #print(cpu_slacks)
    ax2.plot(cpu_x[skip:], cpu_slacks, 'b--', linewidth=2, label='CPU slack')
    ax2.plot(cpu_x[skip:], vpa_slacks, 'g-', linewidth=2, label='VPA slack')
    leg2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg2_lines = leg2.get_lines()
    plt.setp(leg2_lines, linewidth=5)
    
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
    leg1 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg1_lines = leg1.get_lines()
    plt.setp(leg1_lines, linewidth=5)

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
    
    #cpu_usages = [70.945423, 65.442296, 64.409863, 119.784497, 135.455987, 132.949178, 136.771872, 175.153478, 185.904079, 183.262895, 192.68772, 227.425753, 254.816212, 254.478113, 260.506071, 303.346596, 308.617227, 299.22515, 305.235282, 311.828772, 337.069025, 341.031472, 343.669081, 328.119776, 338.280342, 344.441163, 344.441163, 335.140099, 350.777963, 349.91699, 340.089565, 377.237572, 374.370458, 376.32487, 361.708703, 406.344532, 405.11971, 405.017317, 408.076594, 408.210802, 406.260407, 406.374084, 407.719058, 406.356785, 406.362214, 406.362214, 404.446423, 387.809218, 408.078899, 408.078899, 372.243456, 372.243456, 376.154602, 377.50961, 376.743622, 349.482621, 349.482621, 347.499424, 347.499424, 336.362507, 345.883619, 337.922989, 341.92927, 331.631114, 337.574917, 332.435077, 332.435077, 325.964683, 307.368739, 311.580147, 301.298436, 252.496212, 270.463215, 256.371774, 265.598395, 205.370252, 194.007955, 190.262442, 195.444643, 139.251378, 142.869633, 135.796605, 112.902592, 66.472793, 66.472793, 66.193411, 67.865584, 66.77803, 66.77803, 70.657887, 67.437995, 68.226009, 67.539034, 66.349509, 64.506136, 66.219697, 67.510551, 67.991665, 67.068337, 66.33974, 62.580748, 65.642412, 66.541548, 70.586391, 64.729692, 67.637338, 68.881708, 67.422404, 64.984688, 65.97749, 64.506194, 69.634008, 68.147372, 66.590158, 67.052963, 67.052963, 71.358626, 68.882216, 69.519414, 69.519414, 70.228999, 74.246052, 68.462505, 73.169358, 70.098157, 65.387362, 72.08145, 71.065703, 67.787508, 67.787508, 66.458999, 68.932161, 68.626819, 66.414441, 69.212387, 74.520871, 64.554352, 64.477423, 69.332949, 68.953759, 71.484296, 64.172944, 62.394221, 70.175386, 68.779043, 66.403721, 67.570241, 144.856782, 144.856782, 143.247413, 140.610964, 167.288596, 194.501547, 190.03789, 189.772435, 215.651063, 265.392033, 251.205724, 269.74442, 283.230117, 283.230117, 307.84315, 306.034056, 316.582364, 338.444161, 346.708794, 337.349865, 337.349865, 344.826399, 343.451073, 344.821114, 349.517567, 351.68738, 347.484378, 337.902327, 376.243902, 378.518296, 377.202081, 377.202081, 407.452107, 407.452107, 405.998908, 404.946085, 392.102185, 408.492917, 407.493804, 408.369267, 407.863674, 406.321005, 409.703717, 391.433347, 406.054267, 406.771528, 404.792209, 386.355485, 377.29304, 377.407228, 377.480392, 379.254246, 344.950444, 348.037006, 352.364188, 348.345395, 323.045971, 339.935917, 347.251705, 343.856943, 298.840484, 331.755164, 343.582499, 320.213632, 300.478595, 307.248907, 307.812206, 291.884867, 261.702297, 259.051078, 262.933378, 256.393071, 182.59464, 190.104381, 189.615586, 184.850591, 146.253126, 141.016502, 142.792523, 131.8115, 131.8115, 68.825431, 66.605583, 67.565538, 63.642538, 66.383205, 64.75593, 67.763187, 70.798345, 66.71584, 66.490961, 66.544909, 71.078104, 63.922763, 67.359739, 67.359739, 68.505196, 67.824338, 67.923622, 65.67874, 68.718988, 68.718988, 69.227808, 68.567626, 66.236974, 66.717442, 70.859004, 64.59916, 71.125389, 71.233049, 71.233049, 67.978006, 69.599998, 65.841129, 65.242505, 70.706964, 66.175486, 65.676652, 68.72556, 65.771229, 68.517979, 70.358395, 66.507317, 66.69204, 67.858524, 66.551129, 64.101733, 71.257589, 71.290914, 68.487113, 68.487113, 69.81107, 69.81107, 71.721427, 65.040494, 66.710092, 68.665638, 66.996037, 62.962659, 62.962659, 69.487323, 67.348318, 64.879687, 64.16536]
    cpu_usages = [0, 56.383222, 64.956819, 74.511054, 137.092105, 144.490632, 135.664184, 191.46086, 193.901668, 231.137622, 254.015658, 257.865274, 257.865274, 292.459281, 308.438162, 334.448989, 350.268369, 316.436642, 337.63748, 326.069912, 326.069912, 347.985385, 350.405846, 374.867632, 374.867632, 401.093578, 404.914935, 389.359708, 400.051693, 401.909594, 408.424633, 408.424633, 387.5732, 403.170383, 379.174711, 376.445244, 373.285204, 349.678897, 351.057496, 328.856445, 338.281372, 336.398735, 322.253146, 327.528024, 321.121557, 306.020918, 305.964783, 292.50045, 263.419859, 263.419859, 193.52459, 188.577358, 194.817514, 140.202515, 136.588266, 139.040622, 83.606168, 68.522077, 65.169742, 69.657027, 69.386602, 69.386602, 65.637872, 67.053758, 66.783656, 68.000242, 65.904767, 68.136969, 66.78044, 67.538351, 69.116381, 66.934356, 72.1687, 69.800479, 67.436403, 67.48608, 67.48608, 69.39241, 66.07788, 68.939634, 66.72333, 67.076674, 67.342869, 70.132456, 67.692827, 67.692827, 68.796467, 62.746049, 63.077078, 65.578514, 65.784275, 64.06972, 69.135028, 66.289714, 63.03415, 65.456886, 63.846081, 64.649757, 68.074699, 68.074699, 67.765493, 64.867269, 68.171587, 63.685002, 67.225631, 63.326328, 68.898277, 67.224461, 63.235752, 66.548302, 68.976773, 65.81827, 66.02423, 64.922527, 66.207772, 68.146729, 68.146729, 69.813872, 72.411748, 66.289293, 67.566671, 69.129082, 65.837947, 65.837947, 66.106472, 64.780006, 71.30697, 65.843414, 65.843414, 67.82024, 68.627559, 67.211805, 64.064805, 67.506278, 68.308985, 67.576534, 64.418038, 70.501597, 68.875575, 65.322449, 66.35341, 68.450842, 68.450842, 68.624278, 68.624278, 66.43181, 66.43181, 67.010819, 90.553244, 140.118802, 137.963402, 173.884587, 187.687369, 237.606474, 260.988916, 249.068926, 289.831193, 301.478977, 286.068991, 332.57716, 330.707269, 316.272557, 340.31749, 329.343071, 346.102866, 346.127074, 335.94874, 375.795281, 375.795281, 368.063001, 406.516139, 404.409192, 384.869114, 402.967267, 406.723255, 406.723255, 389.345584, 403.754244, 402.927221, 357.469571, 370.866678, 347.400352, 347.400352, 345.606059, 330.352303, 337.559347, 327.610647, 339.992017, 339.992017, 319.010257, 299.659498, 305.589023, 259.337236, 258.338037, 245.234324, 187.753289, 189.610383, 164.360193, 134.079564, 127.996954, 64.613255, 67.41163, 71.345325, 64.107849, 71.106749, 67.188524, 68.051222, 68.118773, 71.866113, 67.025597, 67.732048, 69.484032, 71.147695, 68.382098, 65.873846, 70.144203, 69.844878, 68.929947, 69.309277, 71.184663, 74.302781, 65.786308, 60.917791, 66.163269, 70.628926, 66.728942, 67.091833, 66.065228, 68.313131, 65.383748, 69.699375, 65.567956, 69.436807, 62.835375, 65.611811, 66.338474, 66.24859, 65.699575, 59.968479, 63.039737, 67.064284, 64.926206, 62.336839, 64.394692, 64.21316, 62.685151, 64.345965, 64.421969, 65.737258, 66.930372, 61.395794, 63.5185, 63.5185, 66.636257, 66.636257, 69.158113, 65.704968, 65.704968, 65.849538, 65.849538, 68.056485, 66.774907, 65.136015, 65.136015, 67.980634, 65.487293, 65.938022, 66.033652, 64.368364, 64.368364, 66.007691, 71.022942, 63.497061, 66.375547, 63.079109, 66.740437, 66.600454, 66.600454, 68.537764, 63.830106, 64.20714, 67.15505, 67.15505, 66.631135, 67.110697, 70.808845, 69.109578, 61.931167, 61.931167, 70.023903, 63.721167, 68.180703, 141.229463, 144.453571, 144.823086, 144.823086, 195.968679, 228.385598, 228.385598, 244.739579, 266.260714, 300.75671, 295.806977, 331.584031, 329.498867, 321.615446, 321.615446, 316.903002, 348.715567, 347.453612, 348.501109, 352.649185, 379.188884, 365.165308, 404.746974, 404.746974, 403.293525, 402.035184, 402.931485, 405.025627, 405.894751, 404.912104, 377.067094, 379.755562, 379.755562, 337.788155, 351.323373, 349.073786, 323.568291, 339.404602, 308.865899, 333.326221, 330.79993, 280.883912, 295.030471, 271.313737, 245.797608, 258.545489, 220.12455, 185.138841, 192.200091, 142.341616, 135.572836, 137.202105, 65.169855, 65.169855, 65.241722, 63.806257, 64.62442, 65.533144, 68.454161, 72.684024, 69.406512, 70.697171, 69.368867, 69.368867, 69.506791, 64.930296, 68.935391, 67.163677, 67.163677, 67.724065, 66.871461, 68.585443, 67.57764, 67.381861, 64.488068, 68.251985, 68.603133, 69.065199, 65.411262, 66.739369, 66.782394, 68.019753, 65.790356, 65.482162, 63.755582, 67.316169, 62.983441, 67.10277, 67.442412, 64.542605, 69.042731, 68.308548, 68.308548, 68.743627, 66.30981, 67.025797, 67.736266, 65.227058, 68.101161, 64.540875, 65.838035, 66.632284, 69.461638, 62.972202, 67.768112, 62.061558, 63.351223, 64.216604, 65.59268, 61.942893, 63.542876, 63.330674, 64.108028, 63.965978, 65.945367, 63.826879, 63.826879, 65.983497, 66.282057, 68.86628, 64.823645, 66.849533, 70.872343, 70.872343, 66.636433, 65.650549, 67.226528, 66.610296, 70.402445, 66.493134, 68.990648, 68.990648, 69.74935, 67.765821, 68.843192, 71.825005, 71.825005, 70.599926, 70.644931, 68.44815, 68.508757, 63.550971, 66.529393, 70.843852, 111.456694, 143.384279, 142.708064, 193.150059, 193.150059, 189.990239, 251.753541, 253.233204, 283.625349, 302.369251, 294.738385, 309.002527, 344.917951, 314.368718, 336.491029, 341.425057, 334.867211, 356.035353, 334.170298, 381.069156, 381.069156, 367.34812, 405.100939, 407.893883, 388.089008, 408.964669, 408.506975, 408.767754, 389.481776, 408.678538, 408.678538, 377.409433, 377.409433, 341.203166, 345.826703, 347.037413, 315.145689, 342.985109, 339.310093, 319.810004, 330.023215, 300.252839, 292.562454, 296.666542, 254.846791, 259.544868, 240.437667, 189.435885, 186.954192, 149.15364, 126.776602, 122.196621, 69.954554, 63.298136, 63.118263, 63.180265, 63.063254, 65.458693, 64.91563, 67.284061, 64.430942, 65.301521, 65.886171, 67.386719, 67.371396, 69.106079, 66.991808, 63.501383, 69.261273, 67.165266, 61.937959, 74.148124, 68.044437, 67.166079, 69.689115, 66.808585, 68.80745, 67.336739, 67.785965, 67.785965, 65.435313, 71.521264, 66.89139, 66.094265, 62.727683, 62.727683, 67.041838, 63.632348, 63.784161, 69.343725, 67.096184, 68.258065, 68.509551, 67.874483, 68.216374, 65.290121, 64.914439, 66.290155, 65.590451, 65.206128, 65.206128, 66.663659, 65.704936, 68.244265, 65.891208, 70.512753, 63.416685, 64.533169, 69.763094, 64.694162, 68.481162, 62.73894, 62.73894, 66.181798, 65.944076, 61.817738, 67.846639, 60.854451, 64.66163, 66.766932, 65.34576, 69.346818, 70.475162, 64.521862, 64.521862, 67.301458, 65.272155, 65.930255, 68.554935, 70.937084, 58.167621, 58.167621, 66.410461, 66.410461, 65.969397, 67.064354, 67.064354, 67.103822, 69.59259, 69.59259, 66.552973, 63.766226, 66.428246, 64.957248, 120.195843, 120.195843, 142.535417, 142.535417, 142.535417, 174.007965, 252.542621, 253.943097, 260.923522, 303.538166, 283.149185, 345.507535, 341.841084, 337.438018, 342.620071, 344.887717, 328.542436, 344.393978, 334.983622, 377.688819, 378.340876, 378.340876, 385.083542, 402.058055, 388.264239, 404.746409, 407.458463, 403.109068, 386.050984, 401.238096, 385.588854, 374.197458, 374.197458, 324.859425, 350.855274, 350.855274, 334.874073, 338.921802, 341.852364, 327.976028, 333.159323, 303.226318, 297.971776, 308.966387, 245.91204, 255.433008, 255.433008, 184.31879, 188.229308, 157.12398, 157.12398, 125.809161, 70.458655, 74.408948, 67.070023, 67.111991, 63.323102, 63.996919, 64.901998, 64.901998, 69.872097, 65.263454, 70.646789, 70.281037, 70.770037, 65.922058, 73.340971, 67.302576, 68.272177, 69.481168, 71.39055, 69.284053, 68.669524, 67.196572, 69.511, 67.155025, 66.161561, 70.918664, 66.87883, 66.87883, 59.852072, 59.852072, 68.923268, 63.323426, 65.148449, 63.645174, 67.03236, 65.252496, 65.252496, 63.346677, 59.164913, 64.989345, 64.686928, 65.625622, 66.400409, 67.74685, 60.933982, 69.319422, 68.289387, 66.225405, 68.525799, 66.983458, 62.756596, 64.847471, 61.190449, 70.378769, 67.196653, 65.120607, 64.465952, 62.597857, 66.619529, 64.303229, 63.825769, 60.877521, 64.113564, 65.727067, 67.589511, 65.948535, 67.377545, 67.377545, 67.871688, 66.229078, 65.617415, 66.368269, 69.999964, 69.251218, 69.39869, 68.617272, 68.038928, 67.285648, 67.285648, 62.062307, 68.633919, 65.148513, 64.992567, 68.788832, 68.651884, 69.843028, 69.825103, 63.923937]

    cpu_usages = cpu_usages[:144*4]
    cpu_usages = cpu_usages 
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
            print("Saving fig", len(pred_targets))

            print("--------------------cpu usage size:", len(cpu_usages))
            # Plot figure 
            plot_main()
            plot_slack()
            
            fig1.savefig("./main"+ args.dep_name +str(len(pred_targets))+".png",bbox_inches='tight')
            fig2.savefig("./slack"+ args.dep_name +str(len(pred_targets))+".png",bbox_inches='tight')

            with open("./output"+ args.dep_name+str(len(pred_targets))+".txt", "a") as f:
                print("------cpu usage size:", len(cpu_usages), file=f)
                print("cpu_slacks =", cpu_slacks.tolist(), file=f)
                print("----------------------------------", file=f)
                print("vpa_slacks =", vpa_slacks.tolist(), file=f)
                print("----------------------------------", file=f)
                print("cpu_usages =", cpu_usages, file=f)
                print("----------------------------------", file=f)
                print("cpu_requests =", cpu_requests, file=f)
                print("----------------------------------", file=f)
                print("vpa_targets =", vpa_targets, file=f)
                print("----------------------------------", file=f)


        
        sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        print("Loop time:", time.time()-loopstart)
        print("Sleep time:", sleeptime)
        # sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        time.sleep(sleeptime)
        


    
    
    
if __name__ == '__main__':
    main()
