from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

from pprint import pprint
import time
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import math
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

fig = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)



config.load_kube_config()
api_client = client.ApiClient()
plotVPA = True
xs = []
ytarget = []
xt = []
yusage = []
ylower = []
yupper = []
yrequest = []

xholt = []
yholt = [np.nan]

rescale_counter = 0
scaleup = 0
downscale = 0
set_best = False





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

#fig, ax1 = plt.subplots()

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


s_len = int(96)
#PARAMETERS
params = {
    "window_future": 5, #HW
    "window_past": 1, #HW
    "HW_percentile": 95, #HW
    "season_len": s_len, #HW
    "history_len": 3*s_len, #HW
    "rescale_buffer": 150, # FIX
    "scaleup_count": 7, #FIX
    "scaledown_count": 7, #FIX
    "scale_down_buffer": 100,
    "scale_up_buffer":50,
    "scale_up_stable":1,
    "scale_down_stable":1,
    "stable_range": 25
}

scale_d_b = [100,125,150,175,200]
scale_u_b = [25,50, 75, 100,150,200]
#scale stable: only scale when prediction has been stable for x steps
scale_u_s = [3,4,5,6]
scale_d_s = [3,4,5,6]
#scale stable range: define what range is stable
stable_range = [25, 50, 75, 100]

def get_best_params(series):
    global params
    global scale_d_b, scale_u_b, scale_u_s, scale_d_s, stable_range
    global set_best
    error = np.inf
    best_params = None
    for a in scale_d_b:
        for b in scale_u_b:
            for c in scale_u_s:
                for d in scale_d_s:
                    for e in stable_range:

                            params["scale_down_buffer"] = a
                            params["scale_up_buffer"] = b
                            params["scale_up_stable"] = c
                            params["scale_down_stable"] = d
                            params["stable_range"] = e
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
                                        if np.max(series[i-params["scale_down_stable"]:i+1])-np.min(series[i-params["scale_down_stable"]:i+1]) < params["stable_range"]:
                                            #print("CPU request wasted")
                                            CPU_request = p + params["rescale_buffer"]
                                            rescale_counter += 1
                                            downscale = 0
                                    elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                                        if np.max(series[i-params["scale_up_stable"]:i+1])-np.min(series[i-params["scale_up_stable"]:i+1]) < params["stable_range"]:
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
                                if sub[v] > 0:
                                    sub[v] = sub[v] * 20
                            MSE = np.square(sub).mean() 
                            
                            # Maximum rescale x times 
                            if rescale_counter < 1000 and MSE < error: 

                                error = MSE
                                yrequest = yrequest_temp
                                best_params = params.copy()
                                set_best = True
                                
                        
    print(error)
    print(best_params)
    return best_params


def animate2(i):
    global xs, ytarget, xt, yusage, ylower, yupper, yrequest, xholt, yholt
    global rescale_counter, scaleup, downscale, params
    global set_best
    
    ax2.clear()
    skip = params["season_len"]*3
    if len(yrequest)>skip:
        
        hw_slack = np.subtract(yrequest[skip:],yusage[skip:])
        vpa_slack = np.subtract(ytarget[skip:],yusage[skip:])

        ax2.plot(xt[skip:], hw_slack, 'ro-', linewidth=4, label='HW slack')
        ax2.plot(xt[skip:], vpa_slack, 'yo-', linewidth=4, label='VPA slack')

        if len(xt) > 0:
            ax2.text(xt[-1], hw_slack[-1], str(hw_slack[-1]), fontdict=None)
            ax2.text(xt[-1], vpa_slack[-1], str(vpa_slack[-1]), fontdict=None)

        fig2.suptitle('Slack', fontsize=25)

        ax2.tick_params(axis="x", labelsize=20) 
        ax2.tick_params(axis="y", labelsize=20) 
        fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20), fancybox=True, shadow=True, ncol=5, fontsize=25)
        ax2.set_xlabel('Time (s)', fontsize=20)
        ax2.set_ylabel('CPU (millicores)', fontsize=20)
        ax2.set_ylim(bottom=-100)
        ax2.set_ylim(top=505)
    
def animate(i):

    global api_client
    
    global xs, ytarget, xt, yusage, ylower, yupper, yrequest, xholt, yholt
    global plotVPA 
    global rescale_counter, scaleup, downscale, params
    global set_best
    
    if plotVPA:
        target, lowerBound, upperBound = get_cpu_vpa(api_client)

    cpu_metrics_server = get_cpu_metrics_server(api_client)

    cpu_requests = get_cpu_requests(client)
    
    if cpu_metrics_server is not None and cpu_requests is not None:

        ax1.clear()
        
        # If getting VPA recommendations
        
        if plotVPA and target is not None:

            target, lowerBound, upperBound = get_recommendations(target, lowerBound, upperBound)
            
            ytarget.append(target)
            ylower.append(lowerBound)
            yupper.append(upperBound)
            xs = range(len(ytarget))
            xs = [i * 15 for i in xs]
            #ax1.plot(xs, yupper, 'k--', linewidth=4, label='VPA bounds')
            #ax1.plot(xs, ylower, 'k--', linewidth=4)
            
            ax1.plot(xs, ytarget, 'm--', linewidth=4, label='VPA target')
            ax1.text(xs[-1], ytarget[-1], str(ytarget[-1]), fontdict=None)
            ax1.text(xs[-1], ylower[-1], int(ylower[-1]), fontdict=None)
            ax1.text(xs[-1], yupper[-1], int(yupper[-1]), fontdict=None)
        else:
            ytarget.append(np.nan)
            ylower.append(np.nan)
            yupper.append(np.nan)

        if cpu_requests.endswith('m'):
            cpu_requests = cpu_requests[:-1]
        cpu_requests = int(cpu_requests)

        cpu_metrics_value = get_cpu_metrics_value(cpu_metrics_server)
        # When rescaling, CPU usage falls to 0 as new pod starts up
        if cpu_metrics_value <= 0 and len(yusage) > 0:
            cpu_metrics_value = yusage[-1]
        yrequest.append(cpu_requests)
        yusage.append(cpu_metrics_value)


        # 15 seconds per new point in
        xt = range(len(yusage))
        xt = [i * 15 for i in xt]


        # Holt-winter prediction
        season_length = params["season_len"]
        history_length = params["history_len"]
        start_time = season_length * 2
        current_step = len(yusage)


        if current_step >= start_time: 

            
            season = math.ceil((current_step+1)/season_length)
            
            history_start_season = season - (params["history_len"]/season_length)
            if history_start_season < 1:
                history_start_season = 1
            
            history_start = (history_start_season-1) * s_len 
            
            n = int(current_step - history_start)
            print("n: ",n)
            model = ExponentialSmoothing(yusage[-n:], trend="add", damped=False, seasonal="add",seasonal_periods=season_length)
            model_fit = model.fit()


            x = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])
            p = np.percentile(x, params["HW_percentile"])
            if p < 0:
                p = 0
            yholt.append(p)
            # yholt.append(model_fit.forecast())
            xholt = range(len(yholt))
            xholt = [i * 15 for i in xholt]


            ax1.text(xholt[-1], yholt[-1], int(yholt[-1]), fontdict=None)
            ax1.plot(xholt, yholt, 'bo-', linewidth=4, label='Holt-winters')



            # rescale
            if set_best:
                CPU_request = cpu_requests - params["rescale_buffer"]
    
                if CPU_request - p > params["scale_down_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
                    print("1")
                    if np.max(yholt[-params["scale_down_stable"]:])-np.min(yholt[-params["scale_down_stable"]:]) < params["stable_range"]:
                        print("2")
                        #print("CPU request wasted")
                        # Only rescale after best params have been set
                        patch(client, p + params["rescale_buffer"], p + params["rescale_buffer"])
                        rescale_counter += 1
                        downscale = 0
                        
                elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                    print("3")
                    if np.max(yholt[-params["scale_up_stable"]:])-np.min(yholt[-params["scale_up_stable"]:]) < params["stable_range"]:
                        print("4")
                        #print("CPU request too low")
                        
                        patch(client, p + params["rescale_buffer"], p + params["rescale_buffer"])
                        rescale_counter += 1
                        scaleup = 0
            scaleup += 1
            downscale += 1


            if current_step % season_length == 0:

                print("---------------------------USAGE--------------------------------")
                print(yusage)
                print("---------------------------VPA--------------------------------")
                print(ytarget)
                print("---------------------------REQUESTED--------------------------------")
                print(yrequest)
                print("---------------------------HOLTWINTER--------------------------------")
                print(yholt)
                
                if len(yholt) >= season_length+start_time:
                    a = yholt[-params["history_len"]:]
                    a = np.nan_to_num(a)
                    params = get_best_params(a)
            
        else:
            yholt.append(np.nan)

        

        # Plot last value text
        ax1.text(xt[-1], yusage[-1], int(yusage[-1]), fontdict=None)
        ax1.text(xt[-1], yrequest[-1], int(yrequest[-1]), fontdict=None)
        

        # Plot lines 
        ax1.plot(xt, yrequest, 'ro-', linewidth=4, label='Requested')
        ax1.plot(xt, yusage, 'go-', linewidth=4, label='CPU usage')
        
        # Set plot title, legend, labels
        fig.suptitle('nginx pod metrics', fontsize=25)

        txt= ("Fixed parameters \n Future window size: " + str(params["window_future"]) + ", Past window size: " + str(params["window_past"]) + ", HW percentile: " 
        + str(params["HW_percentile"])+ ", Season length: " + str(params["season_len"])+ ", History length: " + str(params["history_len"])+ ", Rescale buffer: " 
        + str(params["rescale_buffer"])+ ", Upscale count: " + str(params["scaleup_count"])+ ", Downscale count: " + str(params["scaledown_count"]))
        fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', size=20)

        
        ax1.tick_params(axis="x", labelsize=20) 
        ax1.tick_params(axis="y", labelsize=20) 
        fig.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20), fancybox=True, shadow=True, ncol=5, fontsize=25)
        ax1.set_xlabel('Time (s)', fontsize=20)
        ax1.set_ylabel('CPU (millicores)', fontsize=20)
        

# spawn a new thread to wait for input 
def get_input():
    global data
    while True:
        data = input()

import threading

def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    global api_client
    global xs, ytarget, xt, yusage, ylower, yupper, yrequest, xholt, yholt
    global plotVPA 
    global data
    plotVPA = True

    data = [None]

    plt.ion()



    input_thread = threading.Thread(target=get_input)
    input_thread.start()

    
    




    patch(client, 500, 500)
    # ani = animation.FuncAnimation(fig, animate, interval=15000)
    # ani1 = animation.FuncAnimation(fig2, animate2, interval=1500)
    #plt.show()

    ax1.set_xlim(left=s_len*2)
    ax2.set_xlim(left=s_len*2)
    fig.set_size_inches(20,12)
    fig2.set_size_inches(20,12)

    starttime = time.time()
    plt.show()
    plt.draw()
    while True:
        
            
            
        animate(1)
        animate2(1)
        
        if data == 'y':
            print("Saving fig")
            plt.show()
            plt.draw()
            fig.savefig("./resultslive/sc"+".png",bbox_inches='tight')
            fig2.savefig("./resultslive/sl"+".png", bbox_inches="tight")  
            plt.pause(0.001)
        sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        time.sleep(sleeptime)
        


    
    
    
if __name__ == '__main__':
    main()
