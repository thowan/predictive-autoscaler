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

    autov1 = client.AutoscalingV1Api()
    corev1 = client.CoreV1Api()
    appsv1 = client.AppsV1Api()
    metricv2 = client.V2beta1PodsMetricStatus()

    count = 300
    w = watch.Watch()


    for event in w.stream(appsv1.list_deployment_for_all_namespaces, label_selector='autoscale=hpa'):
        print("Event: %s %s" % (event['type'], event['object'].metadata.name))
        count -= 1
        # print(event['object'].metadata.labels.get("autoscale"))
        if event['type'] == "ADDED":

            deployment = event['object'].metadata.name
            
            
            body = init_hpa(deployment)

            try:
                api_response = autov1.create_namespaced_horizontal_pod_autoscaler("default", body)
                pprint(api_response)
            except ApiException as e:
                print("Exception when calling AutoscalingV1Api->create_namespaced_horizontal_pod_autoscaler: %s\n" % e)
            if not count:
                w.stop()
    print("Finished stream.")


if __name__ == '__main__':
    main()