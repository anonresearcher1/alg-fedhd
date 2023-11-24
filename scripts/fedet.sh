#!/bin/sh

for fname in '3clusters_Hetero_20E_lr0.001_adam_1eKL_lr1e-5_adam_T3_publicImageNet_wd5e-5_exp1'
do
#     dir='../results_fedkd/fedavg/cifar10'
#     if [ ! -e $dir ]; then
#     mkdir -p $dir
#     fi
    
    python ../main.py \
    --ntrials=2 \
    --rounds=60 \
    --local_ep=20 \
    --local_bs=64 \
    --optim='adam' \
    --lr=0.001 \
    --local_wd=5e-5 \
    --distill_lr=0.00001 \
    --distill_wd=5e-5 \
    --distill_E=1 \
    --distill_T=3 \
    --distill_data=cifar100 \
    --dataset=cifar10 \
    --momentum=0.9 \
    --p_train=1.0 \
    --partition='niid-labeldir' \
    --clustering_setting='3_clusters' \
    --arch_family='hetero' \
    --datadir='../../data/' \
    --logdir='../results_feddfmh/3clusters/' \
    --log_filename=$fname \
    --alg='fedet' \
    --iid_beta=0.5 \
    --niid_beta=0.1 \
    --seed=2023 \
    --gpu=1 \
    --print_freq=10
done 
