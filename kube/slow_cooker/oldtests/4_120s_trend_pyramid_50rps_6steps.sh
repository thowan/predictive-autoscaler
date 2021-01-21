#!/bin/bash


counter=1
while [ $counter -le 6 ]
do
    let "it = $counter * 5"
    echo $it
    echo Put load $it load 120s
    go run main.go -qps $it -concurrency 10 -iterations 12 http://172.16.16.221
    # sleep 1
    ((counter++))
    
done
trap 'exit' INT

counter=6
while [ $counter -ge 1 ]
do
    let "it = $counter * 5"
    echo $it
    echo Put load $it load 120s
    go run main.go -qps $it -concurrency 10 -iterations 12 http://172.16.16.221
    # sleep 1
    ((counter--))
    
done
trap 'exit' INT


