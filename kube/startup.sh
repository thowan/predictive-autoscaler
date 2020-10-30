#!/bin/bash

echo "Starting vagrant K8s"

SCRIPT_ROOT=$(dirname ${BASH_SOURCE})/

cd kubernetes/vagrant-provisioning/
vagrant up
