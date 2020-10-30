# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Uses watch to print the stream of events from list namespaces and list pods.
The script will wait for 10 events related to namespaces to occur within
the `timeout_seconds` threshold and then move on to wait for another 10 events
related to pods to occur within the `timeout_seconds` threshold.
"""

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

from pprint import pprint
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

global api_client
global xs, ytarget, xt, yt, ylower, yupper, yrequest
global plotVPA

def get_cpu_vpa(api_client):
    

    try:
        # HARDCODED vpa name
        ret_metrics = api_client.call_api('/apis/autoscaling.k8s.io/v1/namespaces/default/verticalpodautoscalers/my-rec-vpa', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
        response = ret_metrics[0].data.decode('utf-8')
        a = json.loads(response)
        target = a["status"]["recommendation"]["containerRecommendations"][0]["target"]["cpu"]
        lowerBound = a["status"]["recommendation"]["containerRecommendations"][0]["lowerBound"]["cpu"]
        upperBound = a["status"]["recommendation"]["containerRecommendations"][0]["upperBound"]["cpu"]
    except:
        return (None,None,None)
    
    return(target, lowerBound, upperBound)

def get_cpu_metrics_server(api_client):
    # ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods/nginx-deployment-67c998fb9b-gmxqz', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    # response = ret_metrics[0].data.decode('utf-8')
    # a = json.loads(response)
    # return(a["containers"][0]["usage"]["cpu"])
    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    for i in range(len(a["items"])):
        # HARDCODED deployment name
        if "nginx-deployment" in a["items"][i]["metadata"]["name"]:
            #print(a["items"][i]["containers"][0]["usage"]["cpu"])
            return a["items"][i]["containers"][0]["usage"]["cpu"]

def get_cpu_requests(client):
    try:
        api_instance = client.CoreV1Api()
        pod_list = api_instance.list_namespaced_pod("default")
        for pod in pod_list.items:
            # HARDCODED deployment name
            if "nginx-deployment" in pod.metadata.name:
                pod_name = pod.metadata.name
        api_response = api_instance.read_namespaced_pod(name=pod_name, namespace='default')
        return(api_response.spec.containers[0].resources.requests["cpu"])
    except ApiException as e:
        print('Found exception in reading the logs')


# Plot settings
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
box = ax1.get_position()
#ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
plt.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)




#fig, ax1 = plt.subplots()

def animate(i):
    # graph_data = open('example.txt','r').read()
    # lines = graph_data.split('\n')
    # for line in lines:
    #     if len(line) > 1:
    #         x, y = line.split(',')
    #         xs.append(float(x))
    #         ytarget.append(float(y))
    global api_client
    
    global xs, ytarget, xt, yt, ylower, yupper, yrequest
    global plotVPA 
    
    if plotVPA:
        target, lowerBound, upperBound = get_cpu_vpa(api_client)

    cpu_metrics_server = get_cpu_metrics_server(api_client)

    cpu_requests = get_cpu_requests(client)
    
    if cpu_metrics_server is not None and cpu_requests is not None:

        ax1.clear()
        
        # If getting VPA recommendations
        
        if plotVPA and target is not None:
            if target.endswith('m'):
                target = target[:-1]
            if lowerBound.endswith('m'):
                lowerBound = lowerBound[:-1]
            if upperBound.endswith('m') or upperBound.endswith('G'):
                upperBound = upperBound[:-1]

            target = int(target)
            lowerBound = int(lowerBound)
            upperBound = int(upperBound)
            ytarget.append(target)
            ylower.append(lowerBound)
            yupper.append(upperBound)
            # ax1.plot(xs, yupper, '--', label='VPA upper bound')
            # ax1.plot(xs, ylower, '--', label='VPA lower bound')
            xs = range(len(ytarget))
            ax1.plot(xs, ytarget, label='VPA target')
            plt.text(xs[-1], ytarget[-1], str(ytarget[-1]), fontdict=None, withdash=False)

        if cpu_requests.endswith('m'):
            cpu_requests = cpu_requests[:-1]
        cpu_requests = int(cpu_requests)


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

        yrequest.append(cpu_requests)
        yt.append(cpu_metrics_value)


        xt = range(len(yt))
        
        
        # Plot last value
        plt.text(xt[-1], yt[-1], int(yt[-1]), fontdict=None, withdash=False)
        plt.text(xt[-1], yrequest[-1], int(yrequest[-1]), fontdict=None, withdash=False)

        # Plot lines 
        ax1.plot(xt, yrequest, label='Requested')
        ax1.plot(xt, yt, label='CPU usage')
        
        # Set plot title, legend, labels
        ax1.title.set_text('nginx pod metrics')
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU (millicores)')
        # ax1.set_ylim(ymin=-0.5)

def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    global api_client
    global xs, ytarget, xt, yt, ylower, yupper, yrequest
    global plotVPA 
    plotVPA = True
    xs = []
    ytarget = []
    xt = []
    yt = []
    ylower = []
    yupper = []
    yrequest = []

    config.load_kube_config()
    api_client = client.ApiClient()

    
    # Patch'
    v1 = client.AppsV1Api()
    dep = {"spec":{"template":{"spec":{"containers":[{"name":"nginx","resources":{"requests":{"cpu":"150m"},"limits":{"cpu":"150m"}}}]}}}}
    resp = v1.patch_namespaced_deployment(name='nginx-deployment',  namespace='default', body=dep)


    print(resp)


    # ani = animation.FuncAnimation(fig, animate, interval=1000)
    # plt.show()
        

if __name__ == '__main__':
    main()