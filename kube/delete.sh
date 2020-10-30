#!/bin/bash

echo "Starting vagrant K8s"

SCRIPT_ROOT=$(dirname ${BASH_SOURCE})/

kubectl delete deployment nginx-deployment 
kubectl apply -f nginx-update.yaml

