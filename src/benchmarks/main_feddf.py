import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.client import *
from src.clustering import *
from src.utils import *

def get_models_feddfmh(num_users, model, dataset, args):

    users_model = []

    for i in range(-1, num_users):
        if model == "resnet8":
            if dataset in ("cifar10"):
                net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=10)
        elif model == "resnet14-0.75":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [1,1,2,2], scaling=0.75, num_classes=10)
        elif model == "resnet14":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [1,2,2,1], scaling=1.0, num_classes=10)
        elif model == "resnet18":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [2,2,2,2], num_classes=10)
        elif model == "resnet34":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [3,4,6,3], num_classes=10)
        elif model == "lenet5": 
            if dataset in ("cifar10"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        else:
            print("not supported yet")
            sys.exit()
        
        if i == -1:
            net_glob = copy.deepcopy(net)
            net_glob.apply(weight_init)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            users_model[i].load_state_dict(initial_state_dict)

    return users_model, net_glob, initial_state_dict

def main_feddf(args):

    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Clients Data')
    
    public_train_ds, public_test_ds, _, \
    _ = get_dataset_global(args.distill_dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0, seed=args.seed)
    
    p_data = torch.utils.data.ConcatDataset([public_train_ds, public_test_ds])
    soft_t = np.random.randn(len(p_data), 10)
    public_ds = DatasetKD(p_data, soft_t)

    train_ds_global, test_ds_global, train_dl_global, \
    test_dl_global = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                        p_train=args.p_test, p_test=args.p_test, seed=args.seed)
    
    train_ds_global1, test_ds_global1, train_dl_global1, \
    test_dl_global1 = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                         p_train=1.0, p_test=1.0, seed=args.seed)
        
    partitions_train, partitions_test, partitions_train_stat, \
    partitions_test_stat = partition_data(args.dataset, args.datadir, args.partition,
                                          args.num_users, niid_beta=args.niid_beta, iid_beta=args.iid_beta,
                                          p_train=args.p_train, p_test=args.p_test, seed=args.seed)

    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    #users_model, net_glob, initial_state_dict = get_models(args, dropout_p=0.5)
    users_model, net_glob, initial_state_dict = get_models_feddfmh(num_users=args.num_users, model=args.model, 
                                                                   dataset=args.dataset, args=args)
    #initial_state_dict = nn.DataParallel(initial_state_dict)
    #net_glob = nn.DataParallel(net_glob)
    print('-'*40)
    print(net_glob)
    print('')

    total = 0
    for name, param in net_glob.named_parameters():
        print(name, param.size())
        total += np.prod(param.size())
        #print(np.array(param.data.cpu().numpy().reshape([-1])))
        #print(isinstance(param.data.cpu().numpy(), np.array))
    print(f'total params {total}')
    print('-'*40)
    ################################# Fixing all to the same Init and data partitioning and random users
    #print(os.getcwd())

    # tt = '../initialization/' + 'partitions_train_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     partitions_train = pickle.load(f)

    # tt = '../initialization/' + 'partitions_train_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     partitions_train = pickle.load(f)

    # tt = '../initialization/' + 'partitions_train_stat_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     partitions_train_stat = pickle.load(f)

    # tt = '../initialization/' + 'partitions_test_stat_'+args.dataset+'_'+args.partition+'.pkl'
    # with open(tt, 'rb') as f:
    #     partitions_test_stat = pickle.load(f)

    #tt = '../initialization/' + 'init_'+args.model+'_'+args.dataset+'.pth'
    #initial_state_dict = torch.load(tt, map_location=args.device)
    #net_glob.load_state_dict(initial_state_dict)

    #server_state_dict = copy.deepcopy(initial_state_dict)
    #for idx in range(args.num_users):
    #    users_model[idx].load_state_dict(initial_state_dict)

    # tt = '../initialization/' + 'comm_users.pkl'
    # with open(tt, 'rb') as f:
    #     comm_users = pickle.load(f)
    ################################# Initializing Clients
    print('-'*40)
    print('Initializing Clients')
    clients = []
    for idx in range(args.num_users):
        sys.stdout.flush()
        print(f'-- Client {idx}, Train Stat {partitions_train_stat[idx]} Test Stat {partitions_test_stat[idx]}')

        noise_level=0
        dataidxs = partitions_train[idx]
        dataidxs_test = partitions_test[idx]

        train_ds_local = get_subset(train_ds_global, dataidxs)
        test_ds_local  = get_subset(test_ds_global, dataidxs_test)

        transform_train, transform_test = get_transforms(args.dataset, noise_level=0, net_id=None, total=0)

        train_dl_local = DataLoader(dataset=train_ds_local, batch_size=args.local_bs, shuffle=True, drop_last=False,
                                   num_workers=4, pin_memory=False)
        test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, num_workers=4,
                                  pin_memory=False)

        clients.append(Client_FedDF(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                   args.lr, args.momentum, args.device, train_dl_local, test_dl_local))

    print('-'*40)
    ###################################### Federation
    print('Starting FL')
    print('-'*40)
    start = time.time()

    if args.new_comer:
        num_users_FL = args.num_users * 4 // 5
        num_users_NC = args.num_users - num_users_FL
    else:
        num_users_FL = args.num_users

    loss_train = []
    clients_local_acc = {i:[] for i in range(num_users_FL)}
    w_locals, loss_locals = [], []
    glob_acc_wavg = []
    glob_acc_kd = []

    w_glob = copy.deepcopy(initial_state_dict)

    m = max(int(args.frac * num_users_FL), 1)

    for iteration in range(args.rounds):

        idxs_users = np.random.choice(range(num_users_FL), m, replace=False)
        #idxs_users = comm_users[iteration]

        print(f'----- ROUND {iteration+1} -----')
        torch.cuda.synchronize()
        sys.stdout.flush()
        for idx in idxs_users:
            clients[idx].set_state_dict(copy.deepcopy(w_glob))

            loss = clients[idx].train(is_print=False)
            loss_locals.append(copy.deepcopy(loss))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))

        ####### FedAvg ####### START
        total_data_points = sum([len(partitions_train[r]) for r in idxs_users])
        fed_avg_freqs = [len(partitions_train[r]) / total_data_points for r in idxs_users]
        w_locals = []
        for idx in idxs_users:
            w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))

        ww = AvgWeights(w_locals, weight_avg=fed_avg_freqs)
        w_glob = copy.deepcopy(ww)
        net_glob.load_state_dict(w_glob)
        _, acc_wavg = eval_test(net_glob, args, test_dl_global1)
        glob_acc_wavg.append(acc_wavg)
        ####### FedAvg ####### END
        
        ###### Logits Avg #######
        logits_locals = []
        for idx in idxs_users:
            logits_locals.append(clients[idx].inference(public_ds))

        teacher_logits = np.mean(logits_locals, axis=0)
        public_ds.set_logits(teacher_logits)
        ###### Logits Avg #######
        
        ##### Global Model KD #####
        net_glob.load_state_dict(copy.deepcopy(w_glob))
        net_glob.to(args.device)
        
        public_dl = torch.utils.data.DataLoader(public_ds, batch_size=128, shuffle=True, drop_last=False)
        steps = int(len(public_ds)/128)
#         optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
        optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.distill_lr)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps, T_mult=1, eta_min=1e-8)
        
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        mse_criterion = nn.MSELoss()
        T = args.distill_T
        for _ in range(args.distill_E):
            batch_loss = []
            for batch_idx, d2 in enumerate(public_dl):
                net_glob.zero_grad()
                #optimizer.zero_grad()
                
                teacher_x, teacher_y, teacher_logits = d2
                teacher_x, teacher_logits = teacher_x.to(args.device), teacher_logits.to(args.device)
                
                logits_student = net_glob(teacher_x)

                kd_loss = kl_criterion(F.log_softmax(logits_student/T, dim=1), F.softmax(teacher_logits/T, dim=1))
                #kd_loss = mse_criterion(F.softmax(logits_student/T, dim=1), F.softmax(teacher_logits/T, dim=1))/2
                loss = (T**2) * kd_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
            
        _, acc_kd = eval_test(net_glob, args, test_dl_global1)
        w_glob = copy.deepcopy(net_glob.state_dict())
        glob_acc_kd.append(acc_kd)
        ##### Global Model KD #####
        
        template = "-- Global Acc Wavg: {:.2f}, After KD: {:.2f}, Global Best Acc: {:.2f}"
        print(template.format(glob_acc_wavg[-1], glob_acc_kd[-1], np.max(glob_acc_kd)))

        print_flag = False
        if iteration+1 in [int(0.10*args.rounds), int(0.25*args.rounds), int(0.5*args.rounds), int(0.8*args.rounds)]:
            print_flag = True

        if print_flag:
            print('*'*25)
            print(f'Check Point @ Round {iteration+1} --------- {int((iteration+1)/args.rounds*100)}% Completed')
            temp_acc = []
            temp_best_acc = []
            for k in range(num_users_FL):
                sys.stdout.flush()
                loss, acc = clients[k].eval_test()
                clients_local_acc[k].append(acc)
                temp_acc.append(clients_local_acc[k][-1])
                temp_best_acc.append(np.max(clients_local_acc[k]))

                template = ("Client {:3d}, current_acc {:3.2f}, best_acc {:3.2f}")
                print(template.format(k, clients_local_acc[k][-1], np.max(clients_local_acc[k])))

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
    temp_acc = []
    temp_best_acc = []

    for k in range(num_users_FL):
        sys.stdout.flush()
        loss, acc = clients[k].eval_test()
        clients_local_acc[k].append(acc)
        temp_acc.append(clients_local_acc[k][-1])
        temp_best_acc.append(np.max(clients_local_acc[k]))

        template = ("Client {:3d}, Final_acc {:3.2f}, best_acc {:3.2f} \n")
        print(template.format(k, clients_local_acc[k][-1], np.max(clients_local_acc[k])))

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))
    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))
    print('*'*25)
    ############################### FedAvg Final Results
    print('-'*40)
    print('FINAL RESULTS')
    template = "-- Global Acc Final: {:.2f}"
    print(template.format(glob_acc_kd[-1]))

    template = "-- Global Acc Avg Final [N*C] Rounds: {:.2f}"
    print(template.format(np.mean(glob_acc_kd[-m:])))

    template = "-- Global Best Acc: {:.2f}"
    print(template.format(np.max(glob_acc_kd)))

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))

    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))

    print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    ############################### FedAvg+ (FedAvg + FineTuning)
    print('-'*40)
    print('FedDF+ ::: FedDF + Local FineTuning')
    sys.stdout.flush()

    local_acc = []
    for idx in range(num_users_FL):
        clients[idx].set_state_dict(copy.deepcopy(w_glob))
        loss = clients[idx].train(is_print=False)
        _, acc = clients[idx].eval_test()
        local_acc.append(acc)

    fedavg_ft_local = np.mean(local_acc)
    print(f'-- FedDF+ :: AVG Local Acc: {np.mean(local_acc):.2f}')
    ############################# Saving Print Results

    ############################# Fairness
    template = ("-- STD of Local Acc: {:3.2f}")
    f1 = np.std(temp_acc)
    print(template.format(f1))

    template = ("-- Top 10% Percentile of Local Acc: {:3.2f}")
    f2 = np.percentile(temp_acc, 90)
    print(template.format(f2))

    template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f}")
    f3 = np.percentile(temp_acc, 10)
    print(template.format(f3))

    template = ("-- Avg Top 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(temp_acc)
    d = int(0.9*num_users_FL)
    f4 = np.mean(np.array(temp_acc)[argsort[d:]])
    print(template.format(f4))

    template = ("-- Avg Bottom 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(temp_acc)
    d = int(0.1*num_users_FL)
    f5 = np.mean(np.array(temp_acc)[argsort[0:d]])
    print(template.format(f5))

    template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
    f6 = f4 - f5
    print(template.format(f6))
    ###########################

    ############################# Fairness
    template = ("-- FedDF+: STD of Local Acc: {:3.2f}")
    ff1 = np.std(local_acc)
    print(template.format(ff1))

    template = ("-- FedDF+: Top 10% Percentile of Local Acc: {:3.2f}")
    ff2 = np.percentile(local_acc, 90)
    print(template.format(ff2))

    template = ("-- FedDF+: Bottom 10% Percentile of Local Acc: {:3.2f}")
    ff3 = np.percentile(local_acc, 10)
    print(template.format(ff3))

    template = ("-- FedDF+: Avg Top 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(local_acc)
    d = int(0.9*num_users_FL)
    ff4 = np.mean(np.array(local_acc)[argsort[d:]])
    print(template.format(ff4))

    template = ("-- FedDF+: Avg Bottom 10% of Local Acc: {:3.2f}")
    argsort = np.argsort(local_acc)
    d = int(0.1*num_users_FL)
    ff5 = np.mean(np.array(local_acc)[argsort[0:d]])
    print(template.format(ff5))

    template = ("-- FedDF+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
    ff6 = ff4 - ff5
    print(template.format(ff6))
    ###########################

    ############################### New Comers Start
    if args.new_comer:
        print('-'*40)
        print('Evaluating new comers')
        sys.stdout.flush()

        new_comer_avg_acc = []
        new_comer_acc = {i:[] for i in range(num_users_FL, args.num_users)}
        for idx in range(num_users_FL, args.num_users):
            clients[idx].set_state_dict(copy.deepcopy(w_glob))
            _, acc = clients[idx].eval_test()
            new_comer_acc[idx].append(acc)
            print(f'Client {idx:3d}, current_acc {acc:3.2f}, best_acc {np.max(new_comer_acc[idx]):3.2f}')
        new_comer_avg_acc.append(np.mean([acc[-1] for acc in new_comer_acc.values()]))
        print(f'-- New Comers Initial AVG Acc: {new_comer_avg_acc[-1]:3.2f}')

        for iteration in range(20):
            for idx in range(num_users_FL, args.num_users):
                loss = clients[idx].train(is_print=False)
                _, acc = clients[idx].eval_test()
                new_comer_acc[idx].append(acc)
            new_comer_avg_acc.append(np.mean([acc[-1] for acc in new_comer_acc.values()]))

            if iteration == 4 or iteration == 9:
                print(f'-- Finetune Round: {iteration + 1}')
                for idx in range(num_users_FL, args.num_users):
                    print(f'Client {idx:3d}, current_acc {new_comer_acc[idx][-1]:3.2f}, best_acc {np.max(new_comer_acc[idx]):3.2f}')
                print(f'-- New Comers AVG Acc: {new_comer_avg_acc[-1]:3.2f}')

        print(f'-- Finetune Finished')
        print(f'-- New Comers Final AVG Acc: {new_comer_avg_acc[-1]:3.2f}')
        print(f'-- New Comers Final Best Acc: {np.mean(new_comer_avg_acc):3.2f}')

    ############################# New Comers End

    final_glob = glob_acc_kd[-1]
    avg_final_glob = np.mean(glob_acc_kd[-m:])
    best_glob = np.max(glob_acc_kd)
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)

    return (final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, fedavg_ft_local, duration,
           f1, f2, f3, f4, f5, f6, ff1, ff2, ff3, ff4, ff5, ff6)


def run_feddf(args, fname):
    alg_name = 'FedDF'

    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_feddf_ft_local=[]
    exp_fl_time=[]
    exp_f1=[]
    exp_f2=[]
    exp_f3=[]
    exp_f4=[]
    exp_f5=[]
    exp_f6=[]
    exp_ff1=[]
    exp_ff2=[]
    exp_ff3=[]
    exp_ff4=[]
    exp_ff5=[]
    exp_ff6=[]

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))

        final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, \
        feddf_ft_local, duration, f1, f2, f3, f4, f5, f6, ff1, ff2, ff3, ff4, ff5, ff6 = main_feddf(args)

        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_feddf_ft_local.append(feddf_ft_local)
        exp_fl_time.append(duration/60)
        exp_f1.append(f1)
        exp_f2.append(f2)
        exp_f3.append(f3)
        exp_f4.append(f5)
        exp_f5.append(f4)
        exp_f6.append(f6)
        exp_ff1.append(ff1)
        exp_ff2.append(ff2)
        exp_ff3.append(ff3)
        exp_ff4.append(ff4)
        exp_ff5.append(ff5)
        exp_ff6.append(ff6)

        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')

        template = "-- Global Final Acc: {:.2f}"
        print(template.format(exp_final_glob[-1]))

        template = "-- Global Avg Final [N*C] Rounds Acc : {:.2f}"
        print(template.format(exp_avg_final_glob[-1]))

        template = "-- Global Best Acc: {:.2f}"
        print(template.format(exp_best_glob[-1]))

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- FedDF+ Fine Tuning Clients AVG Local Acc: {exp_feddf_ft_local[-1]:.2f}')
        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')

        template = ("-- STD of Local Acc: {:3.2f}")
        print(template.format(exp_f1[-1]))

        template = ("-- Top 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_f2[-1]))

        template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_f3[-1]))

        template = ("-- Avg Top 10% of Local Acc: {:3.2f}")
        print(template.format(exp_f4[-1]))

        template = ("-- Avg Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_f5[-1]))

        template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_f6[-1]))

        template = ("-- FedDF+: STD of Local Acc: {:3.2f}")
        print(template.format(exp_ff1[-1]))

        template = ("-- FedDF+: Top 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_ff2[-1]))

        template = ("-- FedDF+: Bottom 10% Percentile of Local Acc: {:3.2f}")
        print(template.format(exp_ff3[-1]))

        template = ("-- FedDF+: Avg Top 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff4[-1]))

        template = ("-- FedDF+: Avg Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff5[-1]))

        template = ("-- FedDF+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f}")
        print(template.format(exp_ff6[-1]))

    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)

    template = "-- Global Final Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)))

    template = "-- Global Avg Final [N*C] Rounds Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)))

    template = "-- Global Best Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)))

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    template = '-- FedDF+ Fine Tuning Clients AVG Local Acc: {:.2f} +- {:.2f}'
    print(template.format(np.mean(exp_feddf_ft_local), np.std(exp_feddf_ft_local)))

    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')

    template = ("-- STD of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f1), np.std(exp_f1)))

    template = ("-- Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f2), np.std(exp_f2)))

    template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f3), np.std(exp_f3)))

    template = ("-- Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f4), np.std(exp_f4)))

    template = ("-- Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f5), np.std(exp_f5)))

    template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_f6), np.std(exp_f6)))

    template = ("-- FedDF+: STD of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff1), np.std(exp_ff1)))

    template = ("-- FedDF+: Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff2), np.std(exp_ff2)))

    template = ("-- FedDF+: Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff3), np.std(exp_ff3)))

    template = ("-- FedDF+: Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff4), np.std(exp_ff4)))

    template = ("-- FedDF+: Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff5), np.std(exp_ff5)))

    template = ("-- FedDF+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_ff6), np.std(exp_ff6)))

    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)

        template = "-- Global Final Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)), file=text_file)

        template = "-- Global Avg Final [N*C] Rounds Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)), file=text_file)

        template = "-- Global Best Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)), file=text_file)

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        template = '-- FedDF+ Fine Tuning Clients AVG Local Acc: {:.2f} +- {:.2f}'
        print(template.format(np.mean(exp_feddf_ft_local), np.std(exp_feddf_ft_local)), file=text_file)

        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)

        template = ("-- STD of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f1), np.std(exp_f1)), file=text_file)

        template = ("-- Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f2), np.std(exp_f2)), file=text_file)

        template = ("-- Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f3), np.std(exp_f3)), file=text_file)

        template = ("-- Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f4), np.std(exp_f4)), file=text_file)

        template = ("-- Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f5), np.std(exp_f5)), file=text_file)

        template = ("-- Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_f6), np.std(exp_f6)), file=text_file)

        template = ("-- FedDF+: STD of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff1), np.std(exp_ff1)), file=text_file)

        template = ("-- FedDF+: Top 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff2), np.std(exp_ff2)), file=text_file)

        template = ("-- FedDF+: Bottom 10% Percentile of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff3), np.std(exp_ff3)), file=text_file)

        template = ("-- FedDF+: Avg Top 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff4), np.std(exp_ff4)), file=text_file)

        template = ("-- FedDF+: Avg Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff5), np.std(exp_ff5)), file=text_file)

        template = ("-- FedDF+: Difference Avg Top and Bottom 10% of Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_ff6), np.std(exp_ff6)), file=text_file)
        print('*'*40)

    return
