#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=feddfmh_cifar10_iid
#SBATCH --err=results/feddfmh_cifar10_iid.err
#SBATCH --out=results/feddfmh_cifar10_iid.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for fname in '3clusters_VGG_20E_lr0.001_adam_1eKL_lr1e-5_adam_T3_publicCifar100_wd5e-5_corrected'
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
    --local_wd=5e-5 \
    --distill_lr=0.00001 \
    --distill_wd=5e-5 \
    --distill_E=1 \
    --distill_T=3 \
    --distill_data=cifar100 \
    --momentum=0.9 \
    --p_train=1.0 \
    --partition='iid' \
    --datadir='../../../data/' \
    --logdir='../../results_feddfmh/3clusters/' \
    --log_filename=$fname \
    --alg='fedmh' \
    --iid_beta=0.5 \
    --niid_beta=0.3 \
    --seed=2023 \
    --gpu=0 \
    --print_freq=10
done 
