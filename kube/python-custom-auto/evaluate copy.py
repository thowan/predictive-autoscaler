import os
import json
import sys
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from pprint import pprint

import math

def main():
    config.load_incluster_config()
    # Parse provided spec into a dict
    spec = json.loads(sys.stdin.read())

    # with open('input2.txt') as json_file:
    #     data = json.load(json_file)
    # print(data)
    evaluate(spec)

def evaluate(spec):
    v1 = client.AppsV1Api()
    replicaset_list = v1.list_namespaced_replica_set("default", label_selector='app=nginx')
    for replicaset in replicaset_list.items:
        if replicaset.status.available_replicas == None:
            continue

        current_replicas = replicaset.metadata.annotations["deployment.kubernetes.io/desired-replicas"]
        #print ("current_replicas:", current_replicas)

    try:
        # HARDCODED target metric value  
        target_value = 20
        value = float(spec["metrics"][0]["value"])
        #print ("metric value:", value)

        target_replicas = math.ceil(int(current_replicas)*value/target_value)
        # HARDCODED min/max replica value
        # if target_replicas < 1:
        #     target_replicas = 1
        # if target_replicas > 5:
        #     target_replicas = 5
        # Build JSON dict with targetReplicas
        evaluation = {}
        evaluation["targetReplicas"] = target_replicas

        # Output JSON to stdout
        sys.stdout.write(json.dumps(evaluation))
    except ValueError as err:
        # If not an integer, output error
        sys.stderr.write(f"Invalid metric value: {err}")
        exit(1)

if __name__ == "__main__":
    main()
