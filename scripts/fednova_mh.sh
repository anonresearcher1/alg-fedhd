#!/bin/sh

for fname in '3clusters_VGG_20E_lr0.01_sgd_wd5e-5_exp1'
do
#     dir='../results_fedkd/fedavg/cifar10'
#     if [ ! -e $dir ]; then
#     mkdir -p $dir
#     fi
    
    python ../main.py \
    --ntrials=2 \
    --rounds=50 \
    --local_ep=20 \
    --local_bs=64 \
    --optim='sgd' \
    --lr=0.01 \
    --momentum=0.9 \
    --local_wd=5e-5 \
    --dataset=cifar100 \
    --p_train=1.0 \
    --partition='niid-labeldir' \
    --clustering_setting='3_clusters' \
    --arch_family='vgg' \
    --datadir='../../data/' \
    --logdir='../results_feddfmh/3clusters/' \
    --log_filename=$fname \
    --alg='fednova_mh' \
    --iid_beta=0.5 \
    --niid_beta=0.1 \
    --seed=2023 \
    --gpu=0 \
    --print_freq=10
done 
