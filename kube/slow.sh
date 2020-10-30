#!/bin/bash

echo "Starting slowcooker with 200 RPS"

cd slow_cooker/
go run main.go -qps 20 -concurrency 10 http://172.16.16.221