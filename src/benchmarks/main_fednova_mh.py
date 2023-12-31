import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import torchvision
import timm
import random 

from src.data import *
from src.models import *
from src.client import *
from src.clustering import *
from src.utils import *

torch.backends.cudnn.benchmark = True

def main_fednova_mh(args):

    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Clients Data')
    
    num_users = [100, 20, 4]
    fracs = [0.1, 0.2, 0.5]
    
    if args.arch_family == 'resnet':
        archs = ['resnet8', 'resnet14', 'resnet18']
    elif args.arch_family == 'vgg':
        archs = ['vgg7', 'vgg11', 'vgg16']
    elif args.arch_family == 'hetero':
        #archs = ['regnetx_002', 'vgg12', 'resnet18']
        #archs = ['edgenext_x_small', 'vgg12', 'resnet18']
        archs = ['vgg7', 'vgg11', 'resnet18']
        
    #num_users = args.num_users_per_cluster
    #archs = args.archs
    #p_trains = args.data_per_cluster
    #fracs = args.frac_per_cluster
    
    print(num_users)
    print(archs)
    print(fracs)
    
    public_train_ds, public_test_ds, _,  \
    _ = get_dataset_global(args.distill_dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0, seed=args.seed)
    
    train_ds_global, test_ds_global, train_dl_global, test_dl_global \
    = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0, seed=args.seed)
    
    Y_train = np.array(train_ds_global.target)
    partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = get_partitions(num_users, train_ds_global, test_ds_global, args)

    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    #print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    
    users_model = []
    net_glob = []
    initial_state_dict = []
    for num, arch in zip(num_users, archs):
        users_model_tmp, net_glob_tmp, initial_state_dict_tmp = get_models_fedmh(num_users=num, model=arch, 
                                                                                dataset=args.dataset, args=args)
        users_model.append(users_model_tmp)
        net_glob.append(net_glob_tmp)
        initial_state_dict.append(initial_state_dict_tmp)
    
    for cn, models in enumerate(users_model): 
        for mod in models: 
            mod.load_state_dict(initial_state_dict[cn])
    
    for cn, mod in enumerate(net_glob): 
        mod.load_state_dict(initial_state_dict[cn])
        
    #initial_state_dict = nn.DataParallel(initial_state_dict)
    #net_glob = nn.DataParallel(net_glob)
    print('-'*40)
    print(net_glob)
    print('')
    
    num_params_list = []
    for cn, mod in enumerate(net_glob): 
        print(f'Model No {cn+1}')
        total = 0
        for name, param in mod.named_parameters():
            print(name, param.size())
            total += np.prod(param.size())
            #print(np.array(param.data.cpu().numpy().reshape([-1])))
            #print(isinstance(param.data.cpu().numpy(), np.array))
        num_params_list.append(total)
        print(f'total params {total}')
        print('-'*40)
        
    print(num_params_list)
    ################################# Initializing Clients
    print('-'*40)
    print('Initializing Clients')
    clients = []
    for cn in range(len(num_users)):
        print(f'---- Clients Group # {cn+1}')
        clients_tmp = []
        for idx in range(num_users[cn]):
            sys.stdout.flush()
            print(f'-- Client {idx}, Train Stat {partitions_train_stat[cn][idx]} Test Stat {partitions_test_stat[cn][idx]}')
        
            noise_level=0
            dataidxs = partitions_train[cn][idx]
            dataidxs_test = partitions_test[cn][idx]
            
            unq, unq_cnt = np.unique(Y_train[dataidxs], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            print(f'Actual Partition Stat: {tmp}')

            train_ds_local = get_subset(train_ds_global, dataidxs)
            test_ds_local  = get_subset(test_ds_global, dataidxs_test)

            bs = min(args.local_bs, len(train_ds_local))
            train_dl_local = DataLoader(dataset=train_ds_local, batch_size=bs, shuffle=True, drop_last=False,
                                    num_workers=4, pin_memory=True)
            test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, num_workers=4,
                                    pin_memory=True)

            if archs[cn] == 'edgenext_x_small':
                optim = 'adamw'
            else:
                optim = args.optim
            print(f'optim: {optim}')
            
            if archs[cn] == 'resnet18':
                #scheduler = True
                scheduler = False
            else:
                scheduler = False
            
            clients_tmp.append(Client_FedNova_MH(idx, copy.deepcopy(users_model[cn][idx]), args.local_bs, args.local_ep, optim,
                    args.lr, args.momentum, args.local_wd, scheduler, args.device, train_dl_local, test_dl_local))
        
        clients.append(clients_tmp)
    print('-'*40)
    ###################################### Federation
    print('Starting FL')
    print('-'*40)
    start = time.time()
    
    loss_train = []
    clients_local_acc = {i:{j:[] for j in range(num_users[i])} for i in range(len(num_users))}
    w_locals, loss_locals = [], []
    glob_acc_wavg = [[] for _ in range(len(num_users))]

    w_glob = copy.deepcopy(initial_state_dict)

    for iteration in range(args.rounds):
        
        idxs_users=[]
        for cn in range(len(num_users)):
            m = max(int(fracs[cn] * num_users[cn]), 1)
            idxs_users_tmp = np.random.choice(range(num_users[cn]), m, replace=False)
            idxs_users.append(idxs_users_tmp)
        
        print(f'----- ROUND {iteration+1} -----')
        torch.cuda.synchronize()
        sys.stdout.flush()
        
        a_list = [[] for _ in range(len(idxs_users))]
        d_list = [[] for _ in range(len(idxs_users))]
        n_list = [[] for _ in range(len(idxs_users))]
        for cn in range(len(idxs_users)):
            for idx in idxs_users[cn]:
                clients[cn][idx].set_state_dict(copy.deepcopy(w_glob[cn]))

                loss, a_i, d_i = clients[cn][idx].train(copy.deepcopy(w_glob[cn]), is_print=False)
                loss_locals.append(copy.deepcopy(loss))

                a_list[cn].append(a_i)
                d_list[cn].append(d_i)
                n_i = len(partitions_train[cn][idx])
                n_list[cn].append(n_i)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))
        
        ####### FedNova ####### START
        for cn in range(len(idxs_users)):
            total_n = sum(n_list[cn])

            d_total_round = copy.deepcopy(initial_state_dict[cn])
            for key in d_total_round:
                d_total_round[key] = torch.zeros_like(initial_state_dict[cn][key])

            for i in range(len(idxs_users[cn])):
                d_para = d_list[cn][i]
                for key in d_para:
                    d_total_round[key] = d_total_round[key].to(args.device) + d_para[key].to(args.device) * n_list[cn][i]/total_n

            # update global model
            coeff = 0.0
            for i in range(len(idxs_users[cn])):
                coeff = coeff + a_list[cn][i] * n_list[cn][i]/total_n

            updated_model = copy.deepcopy(w_glob[cn])
            for key in updated_model:
                d_total_round[key].to(args.device)
                updated_model[key].to(args.device)
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    updated_model[key] = updated_model[key].to(args.device) - (coeff * d_total_round[key]).to(args.device)

            w_glob[cn] = copy.deepcopy(updated_model)
            net_glob[cn].load_state_dict(copy.deepcopy(w_glob[cn]))

            _, acc = eval_test(net_glob[cn], args, test_dl_global)
            glob_acc_wavg[cn].append(acc)
        ####### FedNova ####### END
        print("-- Results:")
        for cn in range(len(num_users)):
            template = "--[Cluster {:.1f}]: {}, Global Acc Wavg: {:.2f}, Best: {:.2f}"
            print(template.format(cn, archs[cn], glob_acc_wavg[cn][-1], 
                                np.max(glob_acc_wavg[cn])))

        print_flag = False
        if iteration+1 in [int(0.5*args.rounds), int(0.8*args.rounds)]:
            print_flag = True

        if print_flag:
            print('*'*25)
            print(f'Check Point @ Round {iteration+1} --------- {int((iteration+1)/args.rounds*100)}% Completed')
            temp_acc = []
            temp_best_acc = []
            for cn in range(len(num_users)):
                for k in range(num_users[cn]):
                    sys.stdout.flush()
                    loss, acc = clients[cn][k].eval_test()
                    clients_local_acc[cn][k].append(acc)
                    temp_acc.append(clients_local_acc[cn][k][-1])
                    temp_best_acc.append(np.max(clients_local_acc[cn][k]))

                    template = ("Client {:3d}, current_acc {:3.2f}, best_acc {:3.2f}")
                    print(template.format(k, clients_local_acc[cn][k][-1], np.max(clients_local_acc[cn][k])))

            #print('*'*25)
            template = ("-- Avg Local Acc: {:3.2f}")
            print(template.format(np.mean(temp_acc)))
            template = ("-- Avg Best Local Acc: {:3.2f}")
            print(template.format(np.mean(temp_best_acc)))
            print('*'*25)

        loss_train.append(loss_avg)

        ## clear the placeholders for the next round
        loss_locals.clear()

        ## calling garbage collector
        gc.collect()

    end = time.time()
    duration = end-start
    print('-'*40)
    ############################### Testing Local Results
    print('*'*25)
    print('---- Testing Final Local Results ----')
    temp_acc = [[] for _ in range(len(num_users))]
    temp_best_acc = [[] for _ in range(len(num_users))]
    for cn in range(len(num_users)):
        for k in range(num_users[cn]):
            sys.stdout.flush()
            loss, acc = clients[cn][k].eval_test()
            clients_local_acc[cn][k].append(acc)
            temp_acc[cn].append(clients_local_acc[cn][k][-1].numpy())
            temp_best_acc[cn].append(np.max(clients_local_acc[cn][k]))

            template = ("Client {:3d}, Final_acc {:3.2f}, best_acc {:3.2f} \n")
            print(template.format(k, clients_local_acc[cn][k][-1], np.max(clients_local_acc[cn][k])))

#     template = ("-- Avg Local Acc: {:3.2f}")
#     temp_acc = np.array(temp_acc).flatten()
#     print(template.format(temp_acc))
#     template = ("-- Avg Best Local Acc: {:3.2f}")
#     temp_best_acc = np.array(temp_best_acc).flatten()
#     print(template.format(np.mean(temp_best_acc)))
    print('*'*25)
    ############################### FedAvg Final Results
    print('-'*40)
    print('FINAL RESULTS')
    for cn in range(len(num_users)):
        template = "-- Global Acc Final Wavg: {:.2f}"
        print(template.format(glob_acc_wavg[cn][-1]))

        template = "-- Global Acc Avg Final 10 Rounds: {:.2f}"
        print(template.format(np.mean(glob_acc_wavg[cn][-10:])))

        template = "-- Global Best Acc: {:.2f}"
        print(template.format(np.max(glob_acc_wavg[cn])))

        template = ("-- Avg Local Acc: {:3.2f}")
        print(template.format(np.mean(temp_acc[cn])))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(np.mean(temp_best_acc[cn])))

        print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    
    final_glob = []
    avg_final_glob = []
    best_glob = []
    for cn in range(len(num_users)):
        final_glob.append(glob_acc_wavg[cn][-1])
        kk = int(num_users[cn]*fracs[cn])
        avg_final_glob.append(np.mean(glob_acc_wavg[cn][-kk:]))
        best_glob.append(np.max(glob_acc_wavg[cn]))
    
    temp_acc = [item for sublist in temp_acc for item in sublist]
    temp_best_acc = [item for sublist in temp_best_acc for item in sublist]
    print(temp_acc)
    print(temp_best_acc)
    
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)

    return (final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, duration)

def run_fednova_mh(args, fname):
    alg_name = 'FedNova-MH'

    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_fl_time=[]

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))

        final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, \
        duration = main_fednova_mh(args)

        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_fl_time.append(duration/60)

        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')

        template = "-- Global Final Acc: {}"
        r = [float(f'{item:.2f}') for item in exp_final_glob[-1]]
        print(template.format(r))

        template = "-- Global Avg Final 10 Rounds Acc : {}"
        r = [float(f'{item:.2f}') for item in exp_avg_final_glob[-1]]
        print(template.format(r))

        template = "-- Global Best Acc: {}"
        r = [float(f'{item:.2f}') for item in exp_best_glob[-1]]
        print(template.format(r))

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')

    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)

    template = "-- Global Final Acc: {} +- {}"
    r1 = [float(f'{item:.2f}') for item in np.mean(exp_final_glob, axis=0)]
    r2 = [float(f'{item:.2f}') for item in np.std(exp_final_glob, axis=0)]
    print(template.format(r1, r2))

    template = "-- Global Avg Final 10 Rounds Acc: {} +- {}"
    r1 = [float(f'{item:.2f}') for item in np.mean(exp_avg_final_glob, axis=0)]
    r2 = [float(f'{item:.2f}') for item in np.std(exp_avg_final_glob, axis=0)]
    print(template.format(r1, r2))

    template = "-- Global Best Acc: {} +- {}"
    r1 = [float(f'{item:.2f}') for item in np.mean(exp_best_glob, axis=0)]
    r2 = [float(f'{item:.2f}') for item in np.std(exp_best_glob, axis=0)]
    print(template.format(r1, r2))

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')

    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)

        template = "-- Global Final Acc: {} +- {}"
        r1 = [float(f'{item:.2f}') for item in np.mean(exp_final_glob, axis=0)]
        r2 = [float(f'{item:.2f}') for item in np.std(exp_final_glob, axis=0)]
        print(template.format(r1, r2), file=text_file)

        template = "-- Global Avg Final 10 Rounds Acc: {} +- {}"
        r1 = [float(f'{item:.2f}') for item in np.mean(exp_avg_final_glob, axis=0)]
        r2 = [float(f'{item:.2f}') for item in np.std(exp_avg_final_glob, axis=0)]
        print(template.format(r1, r2), file=text_file)

        template = "-- Global Best Acc: {} +- {}"
        r1 = [float(f'{item:.2f}') for item in np.mean(exp_best_glob, axis=0)]
        r2 = [float(f'{item:.2f}') for item in np.std(exp_best_glob, axis=0)]
        print(template.format(r1, r2), file=text_file)

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)

        print('*'*40)

    return