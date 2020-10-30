#!/bin/bash

dou=1

while [ $dou -le 10 ]
SECONDS=0
do
    
    trap 'exit' INT
    sleep 5
    elapsedseconds=$SECONDS
    echo $elapsedseconds
done
trap 'exit' INT


