#!/bin/bash



counter=1
while [ $counter -le 10 ]
do
    let "it = $counter * 5"
    echo $it
    echo "Put load 400RPS 120s"
    go run main.go -qps $it -concurrency 10 -iterations 12 http://172.16.16.221
    # sleep 1
    ((counter++))
    
done
trap 'exit' INT

# while true

# do 
#     # echo "Put load 200RPS 240s"
#     # go run main.go -qps 20 -concurrency 10 -iterations 24 http://172.16.16.221

#     echo "Put load 400RPS 120s"
#     go run main.go -qps 40 -concurrency 10 -iterations 12 http://172.16.16.221&

#     sleep 120
#     # echo "Put load 400RPS 120s"
#     # go run main.go -qps 1 -concurrency 10 -iterations 12 http://172.16.16.221

#     # echo "Put load 600RPS 120s"
#     # go run main.go -qps 10 -concurrency 10 -iterations 12 http://172.16.16.221

#     # echo "Put load 400RPS 120s"
#     # go run main.go -qps 40 -concurrency 10 -iterations 12 http://172.16.16.221

# done

# trap 'exit' INT
