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

def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    config.load_kube_config()

    v1 = client.CoreV1Api()
    count = 300
    w = watch.Watch()

    # pod_list = v1.list_namespaced_pod("default")
    # for pod in pod_list.items:
    #     print("%s\t%s\t%s" % (pod.metadata.name, 
    #                         pod.status.phase,
    #                         pod.status.pod_ip))

    # stream = w.stream(v1.list_namespaced_pod, "default")
    # for event in stream:
    #     print("Event: %s %s" % (event['type'], event['object'].metadata.name))

    api_client = client.ApiClient()
    ret_metrics = api_client.call_api('/apis/autoscaling.k8s.io/v1/namespaces/default/verticalpodautoscalers/my-rec-vpa', 'GET', auth_settings = ['BearerToken'], response_type='json', _preload_content=False) 
    
    response = ret_metrics[0].data.decode('utf-8')
    a = json.loads(response)
    print(a["status"]["recommendation"]["containerRecommendations"][0]["target"])
        

if __name__ == '__main__':
    main()