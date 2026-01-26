#!/bin/bash
mkdir -p logs

./run_smooth_adam.sh 1 window 0.1 > logs/window_0.1_adam.log 2>&1 &
./run_smooth_adam.sh 2 window 0.3 > logs/window_0.3_adam.log 2>&1 &
./run_smooth_adam.sh 3 window 0.05 > logs/window_0.05_adam.log 2>&1 &
./run_smooth_adam.sh 4 laplacian 0.1 > logs/laplacian_0.1_adam.log 2>&1 &
./run_smooth_adam.sh 5 laplacian 0.2 > logs/laplacian_0.2_adam.log 2>&1 &
./run_smooth_adam.sh 6 ema 0.5 0.3 > logs/ema_0.3_adam.log 2>&1 &
./run_smooth_adam.sh 0 ema 0.5 0.05 > logs/ema_0.1_adam.log 2>&1 &
./run_smooth_adam.sh 0 ema 0.5 0.1 > logs/ema_0.1_adam.log 2>&1 &

wait
echo "All runs complete"