# FedHD Implementation

This is a [PyTorch](https://pytorch.org/) implementation of FedHD method and the evaluation baselines.
## Usage

We provide scripts to run the algorithms, which are put under `scripts/`. Here is an example to run the script:
```
cd scripts
bash fed.sh
```

The descriptions of parameters are as follows:
| Parameter | Description |
| --------- | ----------- |
| ntrials      | The number of total runs. |
| rounds       | The number of communication rounds per run. |
| num_users    | The number of clients. |
| frac         | `TODO` |
| local_ep     | The number of local training epochs. |
| local_bs     | Local batch size. |
| lr           | The learning rate for local models. |
| momentum     | The momentum for the optimizer. |
| model        | Network architecture. Options: `TODO` |
| dataset      | The dataset for training and testing. Options are discussed above. |
| partition    | How datasets are partitioned. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns). |
| datadir      | The path of datasets. |
| logdir       | The path to store logs. |
| log_filename | The folder name for multiple runs. E.g., with `ntrials=3` and `log_filename=$trial`, the logs of 3 runs will be located in 3 folders named `1`, `2`, and `3`. |
| alg          | Federated learning algorithm. Options are discussed above. |
| beta         | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |
| local_view   | `TODO` |
| noise        | The maximum variance of Gaussian noise added to local party `TODO: what's loacl party?` |
| gpu          | The IDs of GPU to use. E.g., `TODO` |
| print_freq   | The frequency to print training logs. E.g., with `print_freq=10`, training logs are displayed every 10 communication rounds. |


## References: 
* Be your Own Teacher, [[Code](https://github.com/luanyunteng/pytorch-be-your-own-teacher/blob/master/train.py)]
* KD, [[Code](https://github.com/JoonyoungYi/KD-pytorch/blob/master/trainer.py)].

