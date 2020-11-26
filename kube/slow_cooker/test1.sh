#!/bin/bash

dou=1

while [ $dou -le 10 ]
SECONDS=0
do
    counter=1
    while [ $counter -le 10 ]
    do
        let "it = $counter * 7"
        echo $it
        echo Put load $it load 60s
        timeout -s SIGINT 60 go run main.go -qps $it -concurrency 100 http://exilis-cloud137-node1:26995
        # sleep 1
        ((counter++))
        
    done
    timeout -s SIGINT 120 go run main.go -qps 70 -concurrency 100 http://exilis-cloud137-node1:26995

    trap 'exit' INT

    counter=10
    while [ $counter -ge 1 ]
    do
        let "it = $counter * 7"
        echo $it
        echo Put load $it load 60s
        timeout -s SIGINT 60 go run main.go -qps $it -concurrency 100 http://exilis-cloud137-node1:26995
        # sleep 1
        ((counter--))
        
    done
    trap 'exit' INT
    timeout -s SIGINT 720 go run main.go -qps $it -concurrency 100 http://exilis-cloud137-node1:26995

    elapsedseconds=$SECONDS
    echo $elapsedseconds
done
trap 'exit' INT


