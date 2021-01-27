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
            # else:
            #     lstm_model = update_lstm(steps_in, steps_out, n_features, np.array(cpu_usages[-params["history_len"]:]), ywindow, lstm_model)
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
    
    cpu_usages = [0, 0, 61.743225, 65.812928, 64.688258, 61.283377, 61.283377, 67.654982, 68.506236, 68.069168, 63.33223, 67.350238, 67.350238, 59.135644, 59.135644, 65.063693, 65.367389, 64.644802, 64.644802, 63.671429, 67.479587, 60.369221, 63.746355, 68.426908, 68.426908, 67.848495, 68.379601, 65.033326, 66.127736, 66.072899, 64.005658, 65.626443, 65.626443, 66.947988, 64.115421, 65.943284, 68.052725, 66.706593, 64.079707, 66.359751, 66.652653, 70.252121, 62.923355, 67.126152, 63.567186, 63.522037, 65.607079, 65.990171, 62.548382, 61.127716, 62.720609, 64.266515, 65.87071, 65.87071, 65.245995, 57.512539, 59.729175, 64.033217, 63.277707, 63.300236, 65.711968, 63.283551, 64.754742, 64.754742, 68.782502, 63.018085, 63.018085, 63.563587, 71.889895, 64.555834, 62.135457, 61.660707, 62.18573, 62.18573, 67.263784, 71.418236, 137.838296, 124.973905, 144.046814, 188.035742, 186.621325, 198.083011, 247.179403, 244.414532, 290.002148, 290.002148, 295.501502, 340.193935, 337.603942, 337.603942, 337.75654, 342.075036, 340.059003, 357.540656, 333.370868, 378.765409, 384.203911, 376.488329, 417.234252, 419.647498, 399.808957, 420.418237, 417.371609, 415.687412, 395.31032, 409.665596, 414.328398, 367.193016, 380.588229, 355.168267, 351.698839, 331.657607, 338.393024, 338.393024, 323.316553, 323.316553, 335.339155, 303.564811, 304.098715, 250.799748, 250.799748, 252.688684, 228.040479, 195.5729, 181.70091, 135.338763, 135.338763, 107.944042, 107.944042, 66.244164, 61.989939, 66.483219, 65.621958, 64.934725, 66.901524, 66.69103, 64.781212, 67.526626, 67.774357, 65.041874, 70.976161, 69.392448, 67.182445, 68.313569, 65.735736, 69.748069, 69.879586, 68.653724, 63.957029, 63.507327, 68.12431, 66.650802, 65.715767, 60.000721, 67.136805, 63.062296, 64.018353, 62.040963, 62.040963, 63.679135, 62.154706, 65.181957, 64.068167, 62.955775, 63.776876, 64.813811, 67.133909, 63.064697, 63.378305, 64.617083, 65.588123, 63.81287, 67.709429, 65.932648, 69.467776, 68.323161, 66.588076, 64.669751, 63.069118, 67.945729, 67.22985, 66.832906, 66.23627, 65.847554, 65.495303, 65.495303, 66.514129, 67.329368, 65.083671, 61.357294, 62.94477, 62.212584, 68.336662, 67.07261, 64.864739, 64.958125, 69.507114, 66.610447, 68.525375, 65.489864, 68.253549, 65.888995, 70.468058, 75.914542, 68.394364, 69.441889, 65.384635, 67.455503, 67.455503, 68.250114, 65.357909, 65.483581, 65.483581, 71.977791, 72.614541, 72.614541, 67.588842, 63.59081, 68.024119, 63.750568, 120.006415, 141.780724, 157.363302, 194.491844, 191.004811, 214.548369, 258.842478, 247.114835, 312.71089, 301.050892, 290.10112, 329.702207, 324.740036, 327.989362, 341.644953, 345.155229, 339.429658, 349.754034, 343.547861, 381.285517, 370.047963, 370.047963, 404.025968, 406.407655, 395.729498, 410.945677, 408.927901, 388.462345, 411.756913, 413.74045, 379.053994, 377.120611, 374.794915, 342.678069, 350.37731, 351.175823, 330.527249, 339.084112, 314.821851, 345.557808, 335.729661, 306.96319, 294.292478, 279.71473, 263.332449, 263.332449, 252.101558, 197.453348, 193.687384, 151.860015, 136.625863, 138.461229, 68.300933, 65.568534, 59.355894, 66.259219, 70.144654, 69.622521, 66.224944, 66.347775, 67.625393, 71.684717, 69.288331, 67.552996, 66.877703, 65.358259, 64.09007, 65.972771, 46.733962, 46.733962, 59.836314, 67.122512, 67.122512, 69.136134, 69.565368, 66.371765, 68.182048, 66.810451, 64.643932, 60.62675, 68.080732, 66.749273, 64.798237, 64.798237, 64.878494, 63.181709, 65.372129, 64.0131, 65.70622, 69.175132, 66.861682, 64.719033, 67.670241, 64.220562, 68.25315, 67.834423, 68.560337, 69.022853, 69.313214, 65.118357, 67.406357, 63.376912, 65.045172, 64.989495, 63.718965, 63.345824, 65.449804, 64.155342, 62.886053, 66.267959, 65.485001, 65.855195, 69.179225, 66.506739, 66.506739, 65.672874, 65.700215, 62.53296, 68.086563, 66.499444, 62.409591, 62.711345, 62.659603, 62.659603, 60.395247, 66.284063, 65.326346, 67.663318, 66.458327, 67.466311, 70.133281, 65.957265, 66.595222, 67.34767, 63.437278, 63.25499, 67.629983, 66.231469, 67.961705, 68.507882, 67.745954, 67.745954, 67.248327, 62.504716, 94.801841, 137.00814, 145.935319, 190.619658, 194.037242, 233.892233, 233.892233, 254.712385, 261.488186, 298.071202, 282.161501, 329.107831, 317.119975, 312.571976, 341.597658, 340.412251, 341.760491, 355.978433, 354.902114, 363.21097, 381.047892, 371.724801, 414.718742, 411.074446, 399.373427, 409.639409, 410.727897, 393.188453, 410.719083, 415.898426, 392.934029, 377.301244, 379.155277, 379.155277, 349.539475, 335.751355, 335.751355, 340.206766, 344.070271, 315.490784, 315.490784, 292.290156, 304.459899, 302.161018, 255.155852, 255.155852, 203.190348, 187.090546, 193.63851, 146.449029, 133.040927, 133.040927, 63.658729, 65.872208, 65.872208, 63.785919, 65.557305, 66.257614, 66.257614, 64.719509, 69.977905, 63.764268, 72.086588, 72.086588, 79.028644, 72.453438, 72.453438, 71.439419, 65.766206, 67.981376, 69.559621, 66.739438, 72.002492, 64.164944, 64.655924, 64.272103, 64.272103, 62.007494, 62.007494, 66.338223, 71.645633, 71.645633, 63.56426, 64.926313, 66.310069, 64.223916, 70.085048, 68.419701, 71.077381, 71.36494, 64.996268, 70.869812, 72.071281, 70.278834, 66.573504, 64.79114, 67.027187, 62.453225, 62.453225, 64.880695, 62.802072, 65.361954, 63.988112, 68.05674, 68.05674, 67.980978, 69.14486, 67.490425, 71.864219, 63.439282, 64.259946, 66.794305, 65.083107, 63.974744, 64.744144, 66.039328, 63.075114, 64.599929, 69.433168, 67.252682, 69.62072, 68.596196, 65.898732, 65.898732, 64.297847, 66.054371, 67.160351, 66.757853, 66.550352, 65.17163, 65.384049, 71.176669, 68.427492, 68.479803, 67.810531, 67.852539, 69.594607, 69.594607, 70.910444, 68.897539, 70.113685, 70.113685, 62.509112, 65.087484, 133.947896, 133.947896, 132.924342, 165.104443, 188.169863, 214.01004, 257.470853, 240.649592, 308.856074, 309.477951, 290.347634, 322.716367, 333.424267, 333.424267, 345.628135, 340.405617, 342.477336, 352.506857, 345.058327, 383.199048, 384.811998, 384.811998, 417.381313, 417.381313, 378.148455, 411.174043, 409.55963, 409.55963, 390.938918, 407.548575, 392.441869, 382.398889, 382.757602, 335.682254, 335.682254, 355.582738, 349.104937, 339.425138, 311.241604, 325.765666, 336.398238, 336.398238, 277.215851, 297.545856, 254.817334, 254.882559, 244.950759, 187.152664, 182.643752, 135.005404, 135.005404, 137.099465, 93.870301, 67.082988, 65.723279, 64.320887, 69.562654, 74.428905, 65.510198, 65.510198, 62.158134, 65.298171, 67.352314, 65.257533, 64.268172, 64.268172, 64.564241, 62.156164, 62.156164, 66.038028, 64.117921, 64.117921, 65.074717, 68.808897, 61.544367, 61.544367, 63.719097, 66.237215, 67.173829, 61.888583, 66.855439, 69.535783, 71.224133, 67.677779, 65.835607, 67.041732, 63.31931, 62.19923, 62.33675, 63.082979, 67.750766, 67.750766, 65.903316, 65.559874, 66.629098, 66.887576, 63.563459, 67.011674, 61.641178, 65.090452, 67.282378, 67.282378, 62.226595, 71.180984, 63.285314, 63.569488, 62.547156, 67.443055, 62.172521, 62.172521, 68.00451, 66.581996, 66.23648, 66.102292, 67.380522, 67.380522, 68.871685, 68.871685, 64.193322, 66.494357, 60.046868, 60.046868, 64.822877, 64.009062, 67.008353, 65.557842, 65.797809, 65.164547, 61.930419, 68.302495, 68.302495, 62.909795, 70.484541, 64.651806, 64.789164, 69.860725, 65.664353, 66.606772, 67.59384, 68.336629, 69.146114, 64.815284, 67.871863, 69.952916, 121.959888, 136.68046, 139.548716, 197.362649, 197.362649, 182.25515, 254.721091, 246.582766, 246.582766, 295.386888, 295.060452, 327.11073, 327.11073, 316.816171, 349.175202, 328.152328, 352.463196, 349.979274, 342.156759, 382.986456, 383.414558, 375.817267, 409.974599, 411.372933, 410.578232, 414.90799, 418.508568, 398.545694, 411.562525, 411.407412, 411.788265, 364.112456, 364.112456, 339.263475, 351.129407, 323.364911, 337.980744, 333.326596, 333.326596, 335.167825, 330.241032, 280.122562, 287.321545, 297.07834, 279.535228, 254.74456, 254.374602, 179.04024, 188.421524, 189.60678, 189.60678, 134.465954, 105.120722, 64.622324, 64.037213, 60.449515, 67.349291, 66.888078, 66.888078, 65.276643, 65.22202, 66.109513, 63.836765, 64.270034, 65.032251, 71.107685, 71.107685, 63.12264]
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
