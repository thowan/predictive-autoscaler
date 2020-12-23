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

yusage_clone = []
yrequest_clone = []

xholt = []
yholt = [np.nan]

yholt_clone = [np.nan]

rescale_counter = 0
scaleup = 0

downscale = 0

scaleup2 = 0

downscale2 = 0
set_best = False





def get_cpu_vpa(api_client, container_name):
    
    
    try:
        ret_metrics = api_client.call_api('/apis/autoscaling.k8s.io/v1/namespaces/default/verticalpodautoscalers/my-rec-vpa', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
        response = ret_metrics[0].data.decode('utf-8')
        a = json.loads(response)
        containers = a["status"]["recommendation"]["containerRecommendations"]
        container_index = 0
        
        for c in range(len(containers)):
            if container_name in containers[c]["containerName"]:
                container_index = c
                break

        target = a["status"]["recommendation"]["containerRecommendations"][container_index]["target"]["cpu"]
        lowerBound = a["status"]["recommendation"]["containerRecommendations"][container_index]["lowerBound"]["cpu"]
        upperBound = a["status"]["recommendation"]["containerRecommendations"][container_index]["upperBound"]["cpu"]
    except:
        return (None,None,None)
    
    return(target, lowerBound, upperBound)

def patch(client, dep_name, container_name, requests, limits):
    # Patch'
    limits = requests+200
    print(requests)
    v1 = client.AppsV1Api()

    #HARDCODED deployment and container names
    dep = {"spec":{"template":{"spec":{"containers":[{"name":container_name,"resources":{"requests":{"cpu":str(int(requests))+"m"},"limits":{"cpu":str(int(limits))+"m"}}}]}}}}
    resp = v1.patch_namespaced_deployment(name=dep_name,  namespace='default', body=dep)
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


def get_cpu_metrics_server(api_client, dep_name, container_name):

    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    pod_name = get_running_pod(client, dep_name, "default")

    for i in range(len(a["items"])):
        if pod_name in a["items"][i]["metadata"]["name"]:
            #print(a["items"][i]["containers"][0]["usage"]["cpu"])

            containers = a["items"][i]["containers"]
            container_index = 0
            
            for c in range(len(containers)):
                
                if container_name in containers[c]["name"]:
                    container_index = c
                    break
            

            try: 
                ret = a["items"][i]["containers"][container_index]["usage"]["cpu"]
            except IndexError:
                return get_cpu_metrics_server(api_client, dep_name, container_name)
            return ret

def get_cpu_requests(client, dep_name, container_name):
    try:
        api_instance = client.CoreV1Api()
        pod_list = api_instance.list_namespaced_pod("default")
        for pod in pod_list.items:
            # HARDCODED deployment name
            if dep_name in pod.metadata.name:
                pod_name = pod.metadata.name
        api_response = api_instance.read_namespaced_pod(name=pod_name, namespace='default')

        containers = api_response.spec.containers
        container_index = 0
        
        for c in range(len(containers)):
            
            if container_name in containers[c].name:
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


s_len = int(72)
#PARAMETERS
params = {
    "window_future": 5, #HW
    "window_past": 1, #HW
    "HW_percentile": 95, #HW
    "season_len": s_len, #HW
    "history_len": 3*s_len, #HW
    "rescale_buffer": 25, # FIX
    "scaleup_count": 5, #FIX
    "scaledown_count": 5, #FIX
    "scale_down_buffer": 100,
    "scale_up_buffer":50,
    "scale_up_stable":1,
    "scale_down_stable":1,
    "stable_range": 25
}

scale_d_b = np.linspace(25, 200,dtype = int, num=8)
scale_u_b = np.linspace(25, 200,dtype = int, num=8)
#scale stable: only scale when prediction has been stable for x steps
scale_u_s = np.linspace(3, 7,dtype = int, num=4)
scale_d_s = np.linspace(3, 7,dtype = int, num=4)
#scale stable range: define what range is stable
stable_range = np.linspace(25, 100,dtype = int, num=4)

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
    global xs, ytarget, xt, yusage, ylower, yupper, yrequest, xholt, yholt, yusage_clone, yrequest_clone, yholt_clone
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
            ax2.text(xt[-1], hw_slack[-1], str(hw_slack[-1]), fontdict=None, withdash=False)
            ax2.text(xt[-1], vpa_slack[-1], str(vpa_slack[-1]), fontdict=None, withdash=False)

        fig2.suptitle('Slack', fontsize=25)

        ax2.tick_params(axis="x", labelsize=20) 
        ax2.tick_params(axis="y", labelsize=20) 
        fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20), fancybox=True, shadow=True, ncol=5, fontsize=25)
        ax2.set_xlabel('Time (s)', fontsize=20)
        ax2.set_ylabel('CPU (millicores)', fontsize=20)
        ax2.set_ylim(bottom=-20)
        ax2.set_ylim(top=505)
    
def animate(i):

    global api_client
    
    global xs, ytarget, xt, yusage, ylower, yupper, yrequest, xholt, yholt, yusage_clone, yrequest_clone, yholt_clone
    global plotVPA 
    global rescale_counter, scaleup, downscale, params
    global scaleup2, downscale2
    global set_best
    
    if plotVPA:
        target, lowerBound, upperBound = get_cpu_vpa(api_client, "nginx")

    cpu_metrics_server = get_cpu_metrics_server(api_client, "nginx-deployment", "nginx")
    cpu_metrics_server2 = get_cpu_metrics_server(api_client, "clone-deployment", "nginx")

    cpu_requests = get_cpu_requests(client, "nginx-deployment", "nginx")
    cpu_requests2 = get_cpu_requests(client, "clone-deployment", "nginx")

    
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
            ax1.plot(xs, yupper, 'k--', linewidth=4, label='VPA bounds')
            ax1.plot(xs, ylower, 'k--', linewidth=4)
            
            ax1.plot(xs, ytarget, 'm--', linewidth=4, label='VPA target')
            ax1.text(xs[-1], ytarget[-1], str(ytarget[-1]), fontdict=None, withdash=False)
            ax1.text(xs[-1], ylower[-1], int(ylower[-1]), fontdict=None, withdash=False)
            ax1.text(xs[-1], yupper[-1], int(yupper[-1]), fontdict=None, withdash=False)
        else:
            ytarget.append(np.nan)
            ylower.append(np.nan)
            yupper.append(np.nan)

        if cpu_requests.endswith('m'):
            cpu_requests = cpu_requests[:-1]
        cpu_requests = int(cpu_requests)

        if cpu_requests2.endswith('m'):
            cpu_requests2 = cpu_requests2[:-1]
        cpu_requests2 = int(cpu_requests2)

        cpu_metrics_value = get_cpu_metrics_value(cpu_metrics_server)
        cpu_metrics_value2 = get_cpu_metrics_value(cpu_metrics_server2)

        

        # When rescaling, CPU usage falls to 0 as new pod starts up
        if cpu_metrics_value <= 0 and len(yusage) > 0:
            cpu_metrics_value = yusage[-1]

        if cpu_metrics_value2 <= 0 and len(yusage_clone) > 0:
            cpu_metrics_value2 = yusage_clone[-1]
        yrequest.append(cpu_requests)
        yrequest_clone.append(cpu_requests2)
        yusage.append(cpu_metrics_value)
        yusage_clone.append(cpu_metrics_value2)



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
            model2 = ExponentialSmoothing(yusage_clone[-n:], trend="add", damped=False, seasonal="add",seasonal_periods=season_length)
            model_fit = model.fit()
            model_fit2 = model2.fit()

            x = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])
            x2 = model_fit2.predict(start=n-params["window_past"],end=n+params["window_future"])

            p = np.percentile(x, params["HW_percentile"])
            p2 = np.percentile(x2, params["HW_percentile"])
            if p < 0:
                p = 0
            if p2 < 0:
                p2 = 0
            yholt.append(p)
            yholt_clone.append(p2)
            # yholt.append(model_fit.forecast())
            xholt = range(len(yholt))
            xholt = [i * 15 for i in xholt]


            # ax1.text(xholt[-1], yholt[-1], int(yholt[-1]), fontdict=None, withdash=False)
            # ax1.plot(xholt, yholt, 'bo-', linewidth=4, label='Holt-winters pod1')

            # ax1.text(xholt[-1], yholt_clone[-1], int(yholt_clone[-1]), fontdict=None, withdash=False)
            # ax1.plot(xholt, yholt_clone, 'ko-', linewidth=4, label='Holt-winters pod2')



            # rescale
            if set_best:
                CPU_request = cpu_requests - params["rescale_buffer"]
                CPU_request2 = cpu_requests2 - params["rescale_buffer"]
    
                if CPU_request - p > params["scale_down_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
                    print("1")
                    if np.max(yholt[-params["scale_down_stable"]:])-np.min(yholt[-params["scale_down_stable"]:]) < params["stable_range"]:
                        print("2")
                        #print("CPU request wasted")
                        # Only rescale after best params have been set
                        patch(client, "nginx-deployment", "nginx", p + params["rescale_buffer"], p + params["rescale_buffer"])
                        rescale_counter += 1
                        downscale = 0
                        
                elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                    print("3")
                    if np.max(yholt[-params["scale_up_stable"]:])-np.min(yholt[-params["scale_up_stable"]:]) < params["stable_range"]:
                        print("4")
                        #print("CPU request too low")
                        
                        patch(client, "nginx-deployment", "nginx", p + params["rescale_buffer"], p + params["rescale_buffer"])
                        rescale_counter += 1
                        scaleup = 0
                
                # clone
                if CPU_request2 - p2 > params["scale_down_buffer"] and scaleup2 > params["scaleup_count"]  and downscale2 > params["scaledown_count"]:
                    print("1")
                    if np.max(yholt_clone[-params["scale_down_stable"]:])-np.min(yholt_clone[-params["scale_down_stable"]:]) < params["stable_range"]:
                        print("2")
                        #print("CPU request wasted")
                        # Only rescale after best params have been set
                        patch(client, "clone-deployment", "nginx", p2 + params["rescale_buffer"], p2 + params["rescale_buffer"])

                        downscale2 = 0
                        
                elif p2 - CPU_request2 > params["scale_up_buffer"] and scaleup2 > params["scaleup_count"]  and downscale2 > params["scaledown_count"]: 
                    print("3")
                    if np.max(yholt_clone[-params["scale_up_stable"]:])-np.min(yholt_clone[-params["scale_up_stable"]:]) < params["stable_range"]:
                        print("4")
                        #print("CPU request too low")
                        
                        patch(client, "clone-deployment", "nginx", p2 + params["rescale_buffer"], p2 + params["rescale_buffer"])

                        scaleup2 = 0
            scaleup += 1
            downscale += 1
            scaleup2 += 1
            downscale2 += 1


            if current_step % season_length == 0:

                print("---------------------------USAGE--------------------------------")
                print(yusage)
                print("---------------------------VPA--------------------------------")
                print(ytarget)
                print("---------------------------REQUESTED--------------------------------")
                print(yrequest)
                print("---------------------------HOLTWINTER--------------------------------")
                print(yholt)

                print("---------------------------USAGE 2--------------------------------")
                print(yusage_clone)
               
                print("---------------------------REQUESTED2--------------------------------")
                print(yrequest_clone)
                print("---------------------------HOLTWINTER2--------------------------------")
                print(yholt_clone)
                
                if len(yholt) >= season_length+start_time:
                    a = yholt[-params["history_len"]:]
                    a = np.nan_to_num(a)
                    params = get_best_params(a)

               
            
        else:
            yholt.append(np.nan)
            yholt_clone.append(np.nan)

        
        
        # Plot last value text
        ax1.text(xt[-1], yusage[-1], int(yusage[-1]), fontdict=None, withdash=False)
        ax1.text(xt[-1], yrequest[-1], int(yrequest[-1]), fontdict=None, withdash=False)

        ax1.text(xt[-1], yusage_clone[-1], int(yusage_clone[-1]), fontdict=None, withdash=False)
        ax1.text(xt[-1], yrequest_clone[-1], int(yrequest_clone[-1]), fontdict=None, withdash=False)
        

        # Plot lines 
        ax1.plot(xt, yrequest, 'ro-', linewidth=4, label='Requested')
        ax1.plot(xt, yusage, 'go-', linewidth=4, label='CPU usage')

        ax1.plot(xt, yrequest_clone, 'co-', linewidth=4, label='Requested 2 ')
        ax1.plot(xt, yusage_clone, 'yo-', linewidth=4, label='CPU usage 2')
        
        sumlist = [sum(x) for x in zip(yrequest_clone, yrequest)]
        ax1.plot(xt, sumlist, 'o-', linewidth=4, label='CPU requested total')

        # Set plot title, legend, labels
        fig.suptitle('Phase shifted nginx pods', fontsize=25)

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
        time.sleep(1)


import threading

def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    global api_client
    global xs, ytarget, xt, yusage, ylower, yupper, yrequest, xholt, yholt, yusage_clone, yrequest_clone, yholt_clone
    global plotVPA 
    global data
    plotVPA = False

    data = [None]

    plt.ion()



    input_thread = threading.Thread(target=get_input)
    input_thread.start()

    
    




    patch(client, "nginx-deployment", "nginx", 350, 350)
    patch(client, "clone-deployment", "nginx", 340, 340)
    # ani = animation.FuncAnimation(fig, animate, interval=15000)
    # ani1 = animation.FuncAnimation(fig2, animate2, interval=1500)
    #plt.show()

    starttime = time.time()
    plt.show()
    plt.draw()
    while True:
        
            
            
        animate(1)
        animate2(1)
        
        if data == 'y':

            plt.draw()
            plt.pause(0.001)
        sleeptime = 15.0 - ((time.time() - starttime) % 15.0)
        time.sleep(sleeptime)
        


    
    
    
if __name__ == '__main__':
    main()