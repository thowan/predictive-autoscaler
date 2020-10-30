#!/bin/bash

dou=1
counter=1


while [ $dou -le 10 ]
do
    while [ $counter -le 10 ]
    do
        let "it = $counter * 5"
        echo $it
        echo Put load $it load 60s
        go run main.go -qps $it -concurrency 10 -iterations 6 http://172.16.16.221
        # sleep 1
        ((counter++))
        
    done
    go run main.go -qps 50 -concurrency 10 -iterations 12 http://172.16.16.221
    trap 'exit' INT

    counter=5
    while [ $counter -ge 3 ]
    do
        let "it = $counter * 10"
        echo $it
        echo Put load $it load 60s
        go run main.go -qps $it -concurrency 10 -iterations 6 http://172.16.16.221
        # sleep 1
        ((counter--))
        
    done
    trap 'exit' INT

    counter=6
    while [ $counter -le 10 ]
    do
        let "it = $counter * 5"
        echo $it
        echo Put load $it load 60s
        go run main.go -qps $it -concurrency 10 -iterations 6 http://172.16.16.221
        # sleep 1
        ((counter++))
        
    done
    trap 'exit' INT

    counter=5
    while [ $counter -ge 1 ]
    do
        let "it = $counter * 10"
        echo $it
        echo Put load $it load 60s
        go run main.go -qps $it -concurrency 10 -iterations 6 http://172.16.16.221
        # sleep 1
        ((counter--))
        
    done
    trap 'exit' INT
done
trap 'exit' INT


