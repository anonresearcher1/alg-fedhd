#!/bin/bash

adaptive_weight_T=(20.0 8.0 2.0)
arch_family="vgg"
a="0.99"

python ../main.py \
    --ntrials=2 \
    --rounds=50 \
    --local_ep=20 \
    --local_bs=64 \
    --optim='adam' \
    --lr=0.001 \
    --local_wd=5e-5 \
    --distill_lr=0.00001 \
    --distill_wd=5e-5 \
    --distill_E=1 \
    --distill_T=3 \
    --distill_data=imagenet \
    --dataset=cifar100 \
    --momentum=0.9 \
    --p_train=1.0 \
    --adaptive_weight_T "${adaptive_weight_T[@]}"\
    --partition='niid-labeldir' \
    --clustering_setting='3_clusters' \
    --arch_family="$arch_family" \
    --datadir='../../data/' \
    --logdir='../results_feddfmh/3clusters/' \
    --log_filename='3clusters_'$arch_family'3_20E_lr0.001_adam_1eKL_lr1e-5_adam_T3_publicImageNet_Tself20_Aself'$a'_wd5e-5_sT_'"${adaptive_weight_T[*]}"'_exp1' \
    --alg='fedmhw_reg' \
    --iid_beta=0.5 \
    --niid_beta=0.1 \
    --a=$a \
    --seed=2023 \
    --gpu=1 \
    --print_freq=10