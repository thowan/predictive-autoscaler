import os
import json
import sys
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from pprint import pprint



def main():
    # Parse spec into a dict
    #spec = json.loads(sys.stdin.read())
    config.load_kube_config()

    with open('input1.txt') as json_file:
        data = json.load(json_file)
    print(data)

    metric(data)
    

def metric(spec):
    # Get metadata from resource information provided
    metadata = spec["resource"]["metadata"]
    # Get labels from provided metdata
    namespace = metadata["namespace"]
    api_client = client.ApiClient()

    # Get all pods with label
    v1 = client.CoreV1Api()
    num_pod = 0
    total_latency = 0
    
    pod_list = v1.list_namespaced_pod(namespace, label_selector='app=nginx')
    for pod in pod_list.items:
        name = pod.metadata.name
        print (name)

        # For all pods with label, get metric
        
        ret_metrics = api_client.call_api('/apis/custom.metrics.k8s.io/v1beta1/namespaces/'+ namespace +'/pods/' + name + '/response_latency_ms_95th', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    


        response = ret_metrics[0].data.decode('utf-8')
        a = json.loads(response)
        
        latency = a["items"][0]["value"]

        value = int(latency[:-1])
        if latency.endswith('n'):
            value = value / 1000000.0
        elif latency.endswith('m'):
            value = value / 1000.0

        # latency recieved 
        if value < 0:
            value = 0
        print(value)

        num_pod += 1
        total_latency += value

    if num_pod > 0:
        avg_latency = total_latency/num_pod
        print("num_pod:", num_pod)
        print("total_latency:", total_latency)
        print("avg_latency:", avg_latency)
    else:
        print("no pods found")
    
    sys.stdout.write(str(avg_latency))
    # if "numPods" in labels:
    #     # If numPods label exists, output the value of the numPods 
    #     # label back to the autoscaler
    #     sys.stdout.write(labels["numPods"])
    # else:
    #     # If no label numPods, output an error and fail the metric gathering
    #     sys.stderr.write("No 'numPods' label on resource being managed")
    #     exit(1)

if __name__ == "__main__":
    main()