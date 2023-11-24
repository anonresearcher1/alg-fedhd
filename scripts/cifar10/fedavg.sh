#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=fedavg_cifar10_iid
#SBATCH --err=results/fedavg_cifar10_iid.err
#SBATCH --out=results/fedavg_cifar10_iid.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for fname in '3clusters_HeteroArch_20E_lr0.001_adam_wd5e-5'
do
#     dir='../results_fedkd/fedavg/cifar10'
#     if [ ! -e $dir ]; then
#     mkdir -p $dir
#     fi
    
    python ../../run/cifar10/main.py \
    --ntrials=2 \
    --rounds=60 \
    --local_ep=20 \
    --local_bs=64 \
    --lr=0.001 \
    --momentum=0.9 \
    --local_wd=5e-5 \
    --p_train=1.0 \
    --partition='niid-labeldir' \
    --datadir='../../../data/' \
    --logdir='../../results_feddfmh/3clusters/' \
    --log_filename=$fname \
    --alg='fedavg' \
    --iid_beta=0.5 \
    --niid_beta=0.3 \
    --seed=2023 \
    --gpu=0 \
    --print_freq=10
done 
