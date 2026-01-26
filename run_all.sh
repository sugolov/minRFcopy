#!/bin/bash
mkdir -p logs

./run_smooth.sh 0 none > logs/none.log 2>&1 &
./run_smooth.sh 1 window 0.1 > logs/window_0.1.log 2>&1 &
./run_smooth.sh 2 window 0.3 > logs/window_0.3.log 2>&1 &
./run_smooth.sh 3 window 0.5 > logs/window_0.5.log 2>&1 &
./run_smooth.sh 4 laplacian 0.1 > logs/laplacian_0.1.log 2>&1 &
./run_smooth.sh 5 laplacian 0.3 > logs/laplacian_0.3.log 2>&1 &
./run_smooth.sh 6 laplacian 0.5 > logs/laplacian_0.5.log 2>&1 &
./run_smooth.sh 7 ema 0.5 0.3 > logs/ema_0.3.log 2>&1 &

wait
echo "All runs complete"