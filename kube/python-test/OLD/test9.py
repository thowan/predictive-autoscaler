from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

from pprint import pprint
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

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
    print(requests)
    v1 = client.AppsV1Api()

    #HARDCODED deployment and container names
    dep = {"spec":{"template":{"spec":{"containers":[{"name":"nginx","resources":{"requests":{"cpu":str(int(requests))+"m"},"limits":{"cpu":str(int(limits))+"m"}}}]}}}}
    resp = v1.patch_namespaced_deployment(name='nginx-deployment',  namespace='default', body=dep)
    print("PATCHED request, limits:", str(int(requests))+"m", str(int(limits))+"m")


def get_cpu_metrics_server(api_client):

    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    for i in range(len(a["items"])):
        # HARDCODED deployment name
        if "nginx-deployment" in a["items"][i]["metadata"]["name"]:


            containers = a["items"][i]["containers"]
            container_index = 0
            
            for c in range(len(containers)):
                
                if "nginx" in containers[c]["name"]:
                    container_index = c
                    break
            
            return a["items"][i]["containers"][container_index]["usage"]["cpu"]

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
    

# Plot settings
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
box = ax1.get_position()
#ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
plt.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

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


s_len = 16*2
#PARAMETERS
params = {
    "window_future": 4, #HW
    "window_past": 0, #HW
    "HW_percentile": 95, #HW
    "season_len": s_len, #HW
    "history_len": 3*s_len, #HW
    "scale_down_buffer": 100,
    "scale_up_buffer":50,
    "scale_up_stable":1,
    "scale_down_stable":1,
    "rescale_buffer": 25, # FIX
    "rescale_max": 5,
    "scaleup_count": 10, #FIX
    "scaledown_count": 10, #FIX
    "stable_range": 50

}

scale_d_b = [75, 100,150,200]
scale_u_b = [75, 100,150,200]
#scale stable: only scale when prediction has been stable for x steps
scale_u_s = [3,4,5,10]
scale_d_s = [3,4,5,10]
#scale stable range: define what range is stable
stable_range = [25, 50, 75 , 100,150]

def get_best_params(series):
    global params
    global scale_d_b, scale_u_b, scale_u_s, scale_d_s, stable_range
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
                                    sub[v] = sub[v] * 30
                            MSE = np.square(sub).mean() 
                            
                            # Maximum rescale x times 
                            if rescale_counter < 100 and MSE < error: 

                                error = MSE
                                yrequest = yrequest_temp
                                best_params = params.copy()
                                
                        
    print(error)
    print(best_params)
    return best_params

def animate(i):

    global api_client
    
    global xs, ytarget, xt, yusage, ylower, yupper, yrequest, xholt, yholt
    global plotVPA 
    global rescale_counter, scaleup, downscale, params
    
    
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
            ax1.plot(xs, yupper, '--', label='VPA upper bound')
            ax1.plot(xs, ylower, '--', label='VPA lower bound')
            
            ax1.plot(xs, ytarget, label='VPA target')
            plt.text(xs[-1], ytarget[-1], str(ytarget[-1]), fontdict=None, withdash=False)
            plt.text(xs[-1], ylower[-1], int(ylower[-1]), fontdict=None, withdash=False)
            #plt.text(xs[-1], yupper[-1], int(yupper[-1]), fontdict=None, withdash=False)
        else:
            ytarget.append(np.nan)
            ylower.append(np.nan)
            yupper.append(np.nan)

        if cpu_requests.endswith('m'):
            cpu_requests = cpu_requests[:-1]
        cpu_requests = int(cpu_requests)

        cpu_metrics_value = get_cpu_metrics_value(cpu_metrics_server)

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
            model = ExponentialSmoothing(yusage[-season_length*history_length:], trend="add", damped=False, seasonal="add",seasonal_periods=season_length)
            model_fit = model.fit()
            if current_step > season_length*history_length:
                current_step = season_length*history_length

            x = model_fit.predict(start=current_step-params["window_past"],end=current_step+params["window_future"])
            p = np.percentile(x, params["HW_percentile"])
            if p < 0:
                p = 0
            yholt.append(p)
            # yholt.append(model_fit.forecast())
            xholt = range(len(yholt))
            xholt = [i * 15 for i in xholt]


            plt.text(xholt[-1], yholt[-1], int(yholt[-1]), fontdict=None, withdash=False)
            ax1.plot(xholt, yholt, label='Holt-winters')



            # rescale
            CPU_request = cpu_requests
 
            if CPU_request - p > params["scale_down_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
                print("1")
                if np.max(yholt[-params["scale_down_stable"]:])-np.min(yholt[-params["scale_down_stable"]:]) < params["stable_range"]:
                    print("2")
                    #print("CPU request wasted")
                    patch(client, p + params["rescale_buffer"], 500)
                    rescale_counter += 1
                    downscale = 0
            elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                print("3")
                if np.max(yholt[-params["scale_up_stable"]:])-np.min(yholt[-params["scale_up_stable"]:]) < params["stable_range"]:
                    print("4")
                    #print("CPU request too low")
                    patch(client, p + params["rescale_buffer"], 500)
                    rescale_counter += 1
                    scaleup = 0
            scaleup += 1
            downscale += 1

            if current_step % season_length == 0 and len(yholt) > season_length+start_time:
                print(yusage)
                a = yholt[-params["history_len"]:]
                a = np.nan_to_num(a)
                params = get_best_params(a)
            
        else:
            yholt.append(np.nan)

        

        # Plot last value text
        plt.text(xt[-1], yusage[-1], int(yusage[-1]), fontdict=None, withdash=False)
        plt.text(xt[-1], yrequest[-1], int(yrequest[-1]), fontdict=None, withdash=False)
        

        # Plot lines 
        ax1.plot(xt, yrequest, label='Requested')
        ax1.plot(xt, yusage, label='CPU usage')
        
        # Set plot title, legend, labels
        ax1.title.set_text('nginx pod metrics')
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU (millicores)')
        


def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    global api_client
    global xs, ytarget, xt, yusage, ylower, yupper, yrequest, xholt, yholt
    global plotVPA 
    plotVPA = True


    
    





    ani = animation.FuncAnimation(fig, animate, interval=15000)
    plt.show()
        

if __name__ == '__main__':
    main()