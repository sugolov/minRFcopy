#!/bin/bash
GPU=${1:-0}
METHOD=${2:-none}
ALPHA=${3:-0.5}
RHO=${4:-0.5}

source .venv/bin/activate
CUDA_VISIBLE_DEVICES=$GPU python rf.py \
    --cifar \
    --smooth $METHOD \
    --smooth_alpha $ALPHA \
    --smooth_rho $RHO \
    --epochs 100 \
    --lr 5e-4 \
    --batch_size 256 \
    --device cuda \
    --wandb_entity hmeng-university-of-toronto \
    --wandb_project minRF \
    --fid_every 5 \
    --seed 42