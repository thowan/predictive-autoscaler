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
# config.load_kube_config()
# api_client = client.ApiClient()
api_client = None

fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

# Record VPA or not
plotVPA = True

vpa_x = []
vpa_target = []
vpa_lower = []
vpa_upper = []

cpu_x = []
cpu_usage = []
cpu_requested = []

pred_x = []
pred_target = [np.nan]

#PARAMETERS
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
#-------------------------------------------------------------------

def get_cpu_vpa(api_client):
    
    
    try:
        ret_metrics = api_client.call_api('/apis/autoscaling.k8s.io/v1/namespaces/default/verticalpodautoscalers/my-rec-vpa', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
        response = ret_metrics[0].data.decode('utf-8')
        a = json.loads(response)
        containers = a["status"]["recommendation"]["containerRecommendations"]
        container_index = 0
        
        for c in range(len(containers)):
            if "nginx" in containers[c]["containerName"]:
                container_index = c
                break

        target = a["status"]["recommendation"]["containerRecommendations"][container_index]["target"]["cpu"]
        lowerBound = a["status"]["recommendation"]["containerRecommendations"][container_index]["lowerBound"]["cpu"]
        upperBound = a["status"]["recommendation"]["containerRecommendations"][container_index]["upperBound"]["cpu"]
    except:
        return (None,None,None)
    
    return(target, lowerBound, upperBound)

def patch(client, requests, limits):
    # Patch'
    limits = requests
    print(requests)
    v1 = client.AppsV1Api()

    #HARDCODED deployment and container names
    dep = {"spec":{"template":{"spec":{"containers":[{"name":"nginx","resources":{"requests":{"cpu":str(int(requests))+"m"},"limits":{"cpu":str(int(limits))+"m"}}}]}}}}
    resp = v1.patch_namespaced_deployment(name='nginx-deployment',  namespace='default', body=dep)
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
        print('Found exception in reading the logs')

def get_cpu_metrics_server(api_client):
    # ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods/nginx-deployment-67c998fb9b-gmxqz', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    # response = ret_metrics[0].data.decode('utf-8')
    # a = json.loads(response)
    # return(a["containers"][0]["usage"]["cpu"])
    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    pod_name = get_running_pod(client, "nginx-deployment", "default")

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
                time.sleep(1.0)
                return get_cpu_metrics_server(api_client)
            return ret

def get_cpu_requests(client):
    try:
        api_instance = client.CoreV1Api()
        pod_list = api_instance.list_namespaced_pod("default")
        for pod in pod_list.items:
            # HARDCODED deployment name
            if "nginx-deployment" in pod.metadata.name:
                pod_name = pod.metadata.name
        api_response = api_instance.read_namespaced_pod(name=pod_name, namespace='default')

        containers = api_response.spec.containers
        container_index = 0
        
        for c in range(len(containers)):
            
            if "nginx" in containers[c].name:
                container_index = c
                break

        return(api_response.spec.containers[container_index].resources.requests["cpu"])
    except ApiException as e:
        print('Found exception in reading the logs')
    
def get_cpu_metrics_value(cpu_metrics_server):
    if cpu_metrics_server.endswith('m'):
        cpu_metrics_value = int(cpu_metrics_server[:-1])
        cpu_metrics_value = cpu_metrics_value
    elif cpu_metrics_server.endswith('n'):
        cpu_metrics_value = int(cpu_metrics_server[:-1])
        cpu_metrics_value /= 1000000.0
    elif cpu_metrics_server.endswith('u'):
        cpu_metrics_value = int(cpu_metrics_server[:-1])
        cpu_metrics_value /= 1000.0
    else:
        # case cpu == 0
        cpu_metrics_value = int(cpu_metrics_server)
    return cpu_metrics_value

def get_recommendations(target, lowerBound, upperBound):
    if target.endswith('m'):
        target = target[:-1]
    if lowerBound.endswith('m'):
        lowerBound = lowerBound[:-1]
    if upperBound.endswith('m') or upperBound.endswith('G'):
        upperBound = upperBound[:-1]

    target = int(target)
    lowerBound = int(lowerBound)
    upperBound = int(upperBound)
    return target, lowerBound, upperBound

def get_input():
    global data
    while True:
        data = input()

def update_slack_plot():
    global vpa_x, vpa_target, cpu_x, cpu_usage, vpa_lower, vpa_upper, cpu_requested, pred_x, pred_target
    global rescale_counter, scaleup, downscale, params
    global set_best
    
    ax2.clear()
    skip = params["season_len"]*3
    if len(cpu_requested)>skip:
        
        hw_slack = np.subtract(cpu_requested[skip:],cpu_usage[skip:])
        vpa_slack = np.subtract(vpa_target[skip:],cpu_usage[skip:])

        ax2.plot(cpu_x[skip:], hw_slack, 'ro-', linewidth=4, label='HW slack')
        ax2.plot(cpu_x[skip:], vpa_slack, 'yo-', linewidth=4, label='VPA slack')

        if len(cpu_x) > 0:
            ax2.text(cpu_x[-1], hw_slack[-1], str(hw_slack[-1]), fontdict=None)
            ax2.text(cpu_x[-1], vpa_slack[-1], str(vpa_slack[-1]), fontdict=None)

        fig2.suptitle('Slack', fontsize=25)

        ax2.tick_params(axis="x", labelsize=20) 
        ax2.tick_params(axis="y", labelsize=20) 
        fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20), fancybox=True, shadow=True, ncol=5, fontsize=25)
        ax2.set_xlabel('Time (s)', fontsize=20)
        ax2.set_ylabel('CPU (millicores)', fontsize=20)
        ax2.set_ylim(bottom=-100)
        ax2.set_ylim(top=505)
    
def update_main_plot():

    global api_client
    
    global vpa_x, vpa_target, cpu_x, cpu_usage, vpa_lower, vpa_upper, cpu_requested, pred_x, pred_target
    global plotVPA 
    global rescale_counter, scaleup, downscale, params
    ax1.plot([1, 2, 3, 4], [1, 4, 9, 16], 'g--', linewidth=1,label='VPA target')
    print ("Hello")
    return
    
    if plotVPA:
        target, lowerBound, upperBound = get_cpu_vpa(api_client)

    cpu_metrics_server = get_cpu_metrics_server(api_client)

    cpu_requests = get_cpu_requests(client)
    
    if cpu_metrics_server is not None and cpu_requests is not None:

        ax1.clear()
        
        # If getting VPA recommendations
        
        if plotVPA and target is not None:

            target, lowerBound, upperBound = get_recommendations(target, lowerBound, upperBound)
            
            vpa_target.append(target)
            vpa_lower.append(lowerBound)
            vpa_upper.append(upperBound)
            vpa_x = range(len(vpa_target))
            vpa_x = [i * 15 for i in vpa_x]
            #ax1.plot(vpa_x, vpa_upper, 'k--', linewidth=4, label='VPA bounds')
            #ax1.plot(vpa_x, vpa_lower, 'k--', linewidth=4)
            
            ax1.plot(vpa_x, vpa_target, 'm--', linewidth=4, label='VPA target')
            ax1.text(vpa_x[-1], vpa_target[-1], str(vpa_target[-1]), fontdict=None)
            ax1.text(vpa_x[-1], vpa_lower[-1], int(vpa_lower[-1]), fontdict=None)
            ax1.text(vpa_x[-1], vpa_upper[-1], int(vpa_upper[-1]), fontdict=None)
        else:
            vpa_target.append(np.nan)
            vpa_lower.append(np.nan)
            vpa_upper.append(np.nan)

        if cpu_requests.endswith('m'):
            cpu_requests = cpu_requests[:-1]
        cpu_requests = int(cpu_requests)

        cpu_metrics_value = get_cpu_metrics_value(cpu_metrics_server)
        # When rescaling, CPU usage falls to 0 as new pod starts up
        if cpu_metrics_value <= 0 and len(cpu_usage) > 0:
            cpu_metrics_value = cpu_usage[-1]
        cpu_requested.append(cpu_requests)
        cpu_usage.append(cpu_metrics_value)


        # 15 seconds per new point in
        cpu_x = range(len(cpu_usage))
        cpu_x = [i * 15 for i in cpu_x]


        # Holt-winter prediction

        start_time = params["season_len"] * 2
        current_step = len(cpu_usage)


        if current_step >= start_time: 

            
            season = math.ceil((current_step+1)/params["season_len"])
            
            history_start_season = season - (params["history_len"]/params["season_len"])
            if history_start_season < 1:
                history_start_season = 1
            
            history_start = (history_start_season-1) * params["season_len"] 
            
            n = int(current_step - history_start)
            print("n: ",n)
            model = ExponentialSmoothing(cpu_usage[-n:], trend="add", seasonal="add",seasonal_periods=params["season_len"])
            model_fit = model.fit()


            x = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])
            p = np.percentile(x, params["HW_percentile"])
            if p < 0:
                p = 0
            pred_target.append(p)
            # pred_target.append(model_fit.forecast())
            pred_x = range(len(pred_target))
            pred_x = [i * 15 for i in pred_x]


            ax1.text(pred_x[-1], pred_target[-1], int(pred_target[-1]), fontdict=None)
            ax1.plot(pred_x, pred_target, 'bo-', linewidth=4, label='Holt-winters')



            # rescale
            if set_best:
                CPU_request = cpu_requests - params["rescale_buffer"]
    
                if CPU_request - p > params["scale_down_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
                    print("1")
                    if np.max(pred_target[-params["scale_down_stable"]:])-np.min(pred_target[-params["scale_down_stable"]:]) < params["stable_range"]:
                        print("2")
                        #print("CPU request wasted")
                        # Only rescale after best params have been set
                        patch(client, p + params["rescale_buffer"], p + params["rescale_buffer"])
                        rescale_counter += 1
                        downscale = 0
                        
                elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                    print("3")
                    if np.max(pred_target[-params["scale_up_stable"]:])-np.min(pred_target[-params["scale_up_stable"]:]) < params["stable_range"]:
                        print("4")
                        #print("CPU request too low")
                        
                        patch(client, p + params["rescale_buffer"], p + params["rescale_buffer"])
                        rescale_counter += 1
                        scaleup = 0
            scaleup += 1
            downscale += 1

            
        else:
            pred_target.append(np.nan)

        

        # Plot last value text
        ax1.text(cpu_x[-1], cpu_usage[-1], int(cpu_usage[-1]), fontdict=None)
        ax1.text(cpu_x[-1], cpu_requested[-1], int(cpu_requested[-1]), fontdict=None)
        

        # Plot lines 
        ax1.plot(cpu_x, cpu_requested, 'ro-', linewidth=4, label='Requested')
        ax1.plot(cpu_x, cpu_usage, 'go-', linewidth=4, label='CPU usage')
        
        # Set plot title, legend, labels
        fig1.suptitle('nginx pod metrics', fontsize=25)

        txt= ("Fixed parameters \n Future window size: " + str(params["window_future"]) + ", Past window size: " + str(params["window_past"]) + ", HW percentile: " 
        + str(params["HW_percentile"])+ ", Season length: " + str(params["season_len"])+ ", History length: " + str(params["history_len"])+ ", Rescale buffer: " 
        + str(params["rescale_buffer"])+ ", Upscale count: " + str(params["scaleup_count"])+ ", Downscale count: " + str(params["scaledown_count"]))
        fig1.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', size=20)

        
        ax1.tick_params(axis="x", labelsize=20) 
        ax1.tick_params(axis="y", labelsize=20) 
        fig1.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20), fancybox=True, shadow=True, ncol=5, fontsize=25)
        ax1.set_xlabel('Time (s)', fontsize=20)
        ax1.set_ylabel('CPU (millicores)', fontsize=20)
        



def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    global api_client
    global vpa_x, vpa_target, cpu_x, cpu_usage, vpa_lower, vpa_upper, cpu_requested, pred_x, pred_target
    global plotVPA 
    global data
    plotVPA = True

    # Keyboard input
    data = [None]
    input_thread = threading.Thread(target=get_input)
    input_thread.start()

# ---------------------------------------------------------------------

    # patch(client, 500, 500) TODO

    # ax1.set_xlim(left=params["season_len"]*2) TODO
    # ax2.set_xlim(left=params["season_len"]*2) TODO
    fig1.set_size_inches(15,8)
    fig2.set_size_inches(15,8)

    starttime = time.time()

    while True:
        
        update_main_plot()
        # update_slack_plot() TODO
        
        if data == 'y' or len(pred_target)%500 == 0:
            print("Saving fig")
            fig1.savefig("./main"+str(len(pred_target))+".png",bbox_inches='tight')
            #fig2.savefig("./slack"+str(len(pred_target))+".png", bbox_inches="tight")  TODO
            #plt.pause(0.001) TODO need?
        sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        time.sleep(sleeptime)
        


    
    
    
if __name__ == '__main__':
    main()
