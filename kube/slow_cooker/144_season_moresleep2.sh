#!/bin/bash

dou=1

timeout -s SIGINT 1080 go run main.go -qps 7 -concurrency 100 http://10.129.9.3:30426
while [ $dou -le 10 ]
SECONDS=0
do
    counter=1
    while [ $counter -le 10 ]
    do
        let "it = $counter * 7"
        echo $it
        echo Put load $it load 60s
        timeout -s SIGINT 40 go run main.go -qps $it -concurrency 100 http://10.129.9.3:30426
        # sleep 1
        ((counter++))
        
    done
    timeout -s SIGINT 60 go run main.go -qps 70 -concurrency 100 http://10.129.9.3:30426

    trap 'exit' INT

    counter=10
    while [ $counter -ge 1 ]
    do
        let "it = $counter * 7"
        echo $it
        echo Put load $it load 60s
        timeout -s SIGINT 40 go run main.go -qps $it -concurrency 100 http://10.129.9.3:30426
        # sleep 1
        ((counter--))
        
    done
    trap 'exit' INT
    timeout -s SIGINT 1300 go run main.go -qps $it -concurrency 100 http://10.129.9.3:30426

    elapsedseconds=$SECONDS
    echo $elapsedseconds
done
trap 'exit' INT


