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
global xs, ys, xt, yt



def init_hpa(deployment):
    hpa = None
    min_replicas = 1
    max_replicas = 5
    cpu_util = 20
    deployment_name = deployment

    # Create target deployment object
    target = client.V1CrossVersionObjectReference(
            api_version="apps/v1",
            kind="Deployment",
            name=deployment_name)

    # Create HPA specs
    hpa_spec = client.V1HorizontalPodAutoscalerSpec(
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target_cpu_utilization_percentage=cpu_util,
            scale_target_ref=target)
    metadata = client.V1ObjectMeta(name=deployment_name)

    # Create HPA
    hpa = client.V1HorizontalPodAutoscaler(
            api_version="autoscaling/v1",
            kind="HorizontalPodAutoscaler",
            spec=hpa_spec,
            metadata=metadata)
    return hpa

def get_cpu_vpa(api_client):
    ret_metrics = api_client.call_api('/apis/autoscaling.k8s.io/v1/namespaces/default/verticalpodautoscalers/my-rec-vpa', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    return()

def get_cpu_metrics_server(api_client):
    # ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods/nginx-deployment-67c998fb9b-gmxqz', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    # response = ret_metrics[0].data.decode('utf-8')
    # a = json.loads(response)
    # return(a["containers"][0]["usage"]["cpu"])
    ret_metrics = api_client.call_api('/apis/metrics.k8s.io/v1beta1/namespaces/default/pods', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    for i in range(len(a["items"])):
        if "nginx-deployment" in a["items"][i]["metadata"]["name"]:
            #print(a["items"][i]["containers"][0]["usage"]["cpu"])
            return a["items"][i]["containers"][0]["usage"]["cpu"]

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
    #         ys.append(float(y))
    global api_client
    global xs, ys, xt, yt

    cpu_vpa = get_cpu_vpa(api_client)
    
    if cpu_vpa.endswith('m'):
        cpu_vpa = cpu_vpa[:-1]
        
    cpu_vpa = int(cpu_vpa)

    cpu_metrics_server = get_cpu_metrics_server(api_client)
    if cpu_metrics_server is not None:

        
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

        ys.append(cpu_vpa)
        xs = range(len(ys))

        yt.append(cpu_metrics_value)
        xt = range(len(yt))
        
        ax1.clear()

        ax1.plot(xs, ys, label='CPU recommendation target (VPA)')
        ax1.plot(xt, yt, label='CPU usage')
        ax1.title.set_text('nginx pod metrics')
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU (millicores)')

def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    global api_client
    global xs, ys, xt, yt

    xs = []
    ys = []
    xt = []
    yt = []

    config.load_kube_config()
    api_client = client.ApiClient()





    ani = animation.FuncAnimation(fig, animate, interval=15000)
    plt.show()
        

if __name__ == '__main__':
    main()