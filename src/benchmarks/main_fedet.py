import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data import *
from src.models import *
from src.client import *
from src.clustering import *
from src.utils import *

torch.backends.cudnn.benchmark = True

def inference(public_ds, net, args):
    public_dl = torch.utils.data.DataLoader(public_ds, batch_size=64, shuffle=False, drop_last=False)
    
    #net.eval()
    outs = []
    for data, _,_,_ in public_dl:
        data = data.to(args.device)
        out = net(data)
        outs.append(out.detach().cpu())

    outputs = torch.cat(outs).numpy()
    return outputs

def main_fedet(args):

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
        #archs = ['vgg7', 'densenet121', 'resnet18']
        #archs = ['edgenext_x_small', 'vgg12', 'resnet18']
        archs = ['vgg7', 'vgg11', 'resnet18']
    
    #num_users = args.num_users_per_cluster
    #archs = args.archs
    #p_trains = args.data_per_cluster
    #fracs = args.frac_per_cluster
    
    print(num_users)
    print(archs)
    print(fracs)
    
    public_train_ds, public_test_ds, _, \
    _ = get_dataset_global(args.distill_dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0, seed=args.seed)
    if args.distill_dataset == "imagenet":
        subset100 = np.array([], dtype='int')
        tar = np.array(public_train_ds.targets)
        np.random.seed(2023)
        labels = np.random.choice(np.arange(1000), size=100, replace=False)
        for i in labels:
            subset100 = np.hstack([subset100, np.where(tar==i)[0][0:500]])
        public_train_ds = torch.utils.data.Subset(public_train_ds, subset100)
        soft_t = np.random.randn(len(public_train_ds), 100)
        hard_t = np.random.randn(len(public_train_ds), 1)
        public_ds = DatasetKD_ET(public_train_ds, logits=soft_t, labels=hard_t)
    elif args.distill_dataset == "cifar100":
        print('CIFAR-100')
        p_data = torch.utils.data.ConcatDataset([public_train_ds, public_test_ds])
        soft_t = np.random.randn(len(p_data), 10)
        hard_t = np.random.randn(len(p_data), 1)
        public_ds = DatasetKD_ET(p_data, logits=soft_t, labels=hard_t)
        print(len(public_ds))
    
    train_ds_global, test_ds_global, train_dl_global, test_dl_global \
    = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0, seed=args.seed)
    
    Y_train = np.array(train_ds_global.target)
    partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = get_partitions(num_users, train_ds_global, test_ds_global, args)

    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    
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
    for cn in range(len(num_users)):
        scale = [num_params_list[i]/num_params_list[cn] for i in range(len(num_users))]
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
                scheduler = True
            else:
                scheduler = False
            
            clients_tmp.append(Client_FedET(idx, copy.deepcopy(users_model[cn][idx]), args.local_bs, args.local_ep, optim,
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
    glob_acc_kd = [[] for _ in range(len(num_users))]

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
        for cn in range(len(idxs_users)):
            for idx in idxs_users[cn]:
                #print(f'cn {cn} \n idx {idx}')
                clients[cn][idx].set_state_dict(copy.deepcopy(w_glob[cn]))

                loss = clients[cn][idx].train(is_print=False)
                loss_locals.append(copy.deepcopy(loss))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))
        ####### FedAvg ####### START
        for cn in range(len(num_users)):
            total_data_points = sum([len(partitions_train[cn][r]) for r in idxs_users[cn]])
            fed_avg_freqs = [len(partitions_train[cn][r]) / total_data_points for r in idxs_users[cn]]
            w_locals = []
            for idx in idxs_users[cn]:
                w_locals.append(copy.deepcopy(clients[cn][idx].get_state_dict()))

            ww = AvgWeights(w_locals, weight_avg=fed_avg_freqs)
            w_glob[cn] = copy.deepcopy(ww)
            #net_glob[cn].load_state_dict(copy.deepcopy(ww))
            #_, acc = eval_test(net_glob[cn], args, test_dl_global)
            #glob_acc_wavg[cn].append(acc)
        ####### FedAvg ####### END
        
        ###### Logits Avg #######
        logits_locals = []
        indd=0
        for cn in range(len(num_users)):
            for idx in idxs_users[cn]:
                logits_locals.append(clients[cn][idx].inference(public_ds))
                indd+=1

        varss = np.array([np.var(mat, axis=1) for mat in logits_locals])
        vars_normalized = varss/np.mean(varss, axis=0)
        weighted_logits = np.sum([vars_normalized[i][:, np.newaxis] * logits_locals[i] for i in range(indd)], axis=0)
        teacher_labels = np.argmax(weighted_logits, axis=1)
        
        public_ds.set_labels(teacher_labels)
        
        teacher_logits = []
        for kk in range(logits_locals[0].shape[0]):
            temp_logits = []
            for jj in range(indd):
                if np.argmax(logits_locals[jj][kk]) != teacher_labels[kk]:
                    temp_logits.append(logits_locals[jj][kk])
            if len(temp_logits) >= 1:
                temp_var = np.var(temp_logits, axis=1)
                temp_var_norm = temp_var/np.mean(temp_var, axis=0)
                temp_weighted_logits = np.sum([temp_var_norm[i]*temp_logits[i] for i in range(len(temp_logits))], axis=0)
            elif len(temp_logits) == 0:
                temp_weighted_logits = np.zeros(10)
            teacher_logits.append(temp_weighted_logits)
        
        teacher_logits = np.array(teacher_logits)
        public_ds.set_logits(teacher_logits)
        ###### Logits Avg #######
        
        ##### Global Model KD #####
        for cn in range(len(num_users)):
            net_glob[cn].load_state_dict(copy.deepcopy(w_glob[cn]))
            net_glob[cn].to(args.device)
        
            public_dl = torch.utils.data.DataLoader(public_ds, batch_size=128, shuffle=True, drop_last=False)
            steps = int(len(public_ds)/128)
            optimizer = torch.optim.Adam(net_glob[cn].parameters(), lr=args.distill_lr, weight_decay=args.distill_wd)
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

            ce_criterion = nn.CrossEntropyLoss()
            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            T = args.distill_T

            for _ in range(args.distill_E):
                batch_loss = []
                for batch_idx, d2 in enumerate(public_dl):
                    net_glob[cn].zero_grad()

                    teacher_x, teacher_y, teacher_logits, teacher_labels = d2
                    teacher_x, teacher_logits = teacher_x.to(args.device), teacher_logits.to(args.device)
                    teacher_labels = teacher_labels.to(args.device)

                    logits_student = net_glob[cn](teacher_x)

                    ce_loss = ce_criterion(logits_student, teacher_labels)
                    kd_loss = kl_criterion(F.log_softmax(logits_student/T, dim=1), F.softmax(teacher_logits/T, dim=1))
                    
                    loss = ce_loss + 0.1 * (T**2) * kd_loss
                    #loss.requires_grad = True
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                #scheduler.step()
                
            w_glob[cn] = copy.deepcopy(net_glob[cn].state_dict())
            _, acc_kd = eval_test(net_glob[cn], args, test_dl_global)
            glob_acc_kd[cn].append(acc_kd)
        ##### Global Model KD #####
        print("-- Results:")
        for cn in range(len(num_users)):
            template = "--[Cluster {:.1f}]: {}, Global Acc KD: {:.2f}, Best: {:.2f}"
            print(template.format(cn, archs[cn], glob_acc_kd[cn][-1], np.max(glob_acc_kd[cn])))
            
        print_flag = False
        if iteration+1 in [int(0.5*args.rounds)]:
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
        template = "-- Global Acc KD: {:.2f}"
        print(template.format(glob_acc_kd[cn][-1]))

        template = "-- Global Acc Avg Final 10 Rounds: {:.2f}"
        print(template.format(np.mean(glob_acc_kd[cn][-10:])))

        template = "-- Global Best Acc: {:.2f}"
        print(template.format(np.max(glob_acc_kd[cn])))

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
        final_glob.append(glob_acc_kd[cn][-1])
        kk = int(num_users[cn]*fracs[cn])
        avg_final_glob.append(np.mean(glob_acc_kd[cn][-kk:]))
        best_glob.append(np.max(glob_acc_kd[cn]))
    
    temp_acc = [item for sublist in temp_acc for item in sublist]
    temp_best_acc = [item for sublist in temp_best_acc for item in sublist]
    print(temp_acc)
    print(temp_best_acc)
    
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)

    return (final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, duration)

def run_fedet(args, fname):
    alg_name = 'FedET'

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
        duration = main_fedet(args)

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