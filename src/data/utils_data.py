import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets 
from PIL import Image
import os
import random
from .datasetzoo import DatasetZoo

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
def get_transforms(dataset, noise_level=0, net_id=None, total=0):
    if dataset == 'mnist':
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    elif dataset == 'usps':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(8, fill=0, padding_mode='constant'),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(8, fill=0, padding_mode='constant'),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset == 'fmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

    elif dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

    elif dataset == 'stl10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif dataset == 'imagenet':
        transform_train = transforms.Compose([
        #transforms.Resize((32, 32)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

        transform_test = transforms.Compose([
            #transforms.Resize((32, 32)),
            #transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    elif dataset == 'femnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, total), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    return transform_train, transform_test

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(np.sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class DatasetKD(Dataset):
    def __init__(self, dataset, logits):
        self.dataset = dataset
        self.logits = logits
    
    def set_logits(self, logits):
        self.logits = logits
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = self.logits[item]
        return image, label, logits
    
class DatasetKD_ET(Dataset):
    def __init__(self, dataset, logits, labels):
        self.dataset = dataset
        self.logits = logits
        self.labels = labels
    
    def set_logits(self, logits):
        self.logits = logits
        
    def set_labels(self, labels):
        self.labels = labels
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = self.logits[item]
        labels = self.labels[item]
        return image, label, logits, labels

class DatasetKD_AE(Dataset):
    def __init__(self, dataset, logits):
        self.dataset = dataset
        self.logits = logits
    
    def set_logits(self, logits):
        self.logits = logits
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = [self.logits[i][item] for i in range(len(self.logits))]
        return image, label, logits
    
class DatasetKD_Self(Dataset):
    def __init__(self, dataset, logits, self_logits):
        self.dataset = dataset
        self.logits = logits
        self.self_logits = self_logits
    
    def set_logits(self, logits):
        self.logits = logits
    
    def set_self_logits(self, logits):
        self.self_logits = logits
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        logits = [self.logits[i][item] for i in range(len(self.logits))]
        self_logits = [self.self_logits[i][item] for i in range(len(self.self_logits))]
        return image, label, logits, self_logits

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
    
def get_subset(dataset, idxs): 
    return DatasetSplit(dataset, idxs)

def get_dataset_global(dataset, datadir, batch_size=128, p_train=1.0, p_test=1.0, seed=2023):
    transform_train, transform_test = get_transforms(dataset, noise_level=0, net_id=None, total=0)
    
    if dataset == "imagenet":
        train_ds_global = datasets.ImageNet(root=datadir+'imagenet_resized/', split='train', transform=transform_train)
        test_ds_global = datasets.ImageNet(root=datadir+'imagenet_resized/', split='val', transform=transform_train)
    else:
        train_ds_global = DatasetZoo(datadir, dataset=dataset, dataidxs=None, train=True, 
                                transform=transform_train, target_transform=None, download=True, p_data=p_train,
                                seed=seed)
    
        test_ds_global = DatasetZoo(datadir, dataset=dataset, dataidxs=None, train=False, 
                                transform=transform_train, target_transform=None, download=True, p_data=p_test,
                                seed=seed)
    
    train_dl_global = DataLoader(dataset=train_ds_global, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dl_global = DataLoader(dataset=test_ds_global, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_ds_global, test_ds_global, train_dl_global, test_dl_global

def dir_partition(num_users, niid_beta=0.5, nclasses=10, y_train=None, y_test=None, train_inds=None):
    idxs_train = np.arange(len(y_train))
    idxs_test = np.arange(len(y_test))

    n_train = y_train.shape[0]

    partitions_train = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_test = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_train_stat = {}
    partitions_test_stat = {}
    
    min_size = 0
    min_require_size = 3
    #np.random.seed(2022)
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(nclasses):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(niid_beta, num_users))
            proportions = np.array([p * (len(idx_j) < n_train/num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        
    #### Assigning samples to each client         
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        partitions_train[j] = np.hstack([partitions_train[j], idx_batch[j]])

        dataidx = partitions_train[j]          
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        partitions_train_stat[j] = tmp

        for key in tmp.keys():
            dataidx = np.where(y_test==key)[0]
            partitions_test[j] = np.hstack([partitions_test[j], dataidx])

        dataidx = partitions_test[j]
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        partitions_test_stat[j] = tmp
        
    for j in range(num_users):
        partitions_train[j] = np.array(train_inds)[partitions_train[j]]
        
    return (partitions_train, partitions_test, partitions_train_stat, partitions_test_stat)


def iid_partition(num_users, nclasses=10, y_train=None, y_test=None, train_inds=None):
    idxs_train = np.arange(len(y_train))
    idxs_test = np.arange(len(y_test))

    n_train = y_train.shape[0]

    partitions_train = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_test = {i:np.array([],dtype='int') for i in range(num_users)}
    partitions_train_stat = {}
    partitions_test_stat = {}
    
    ind2label = {cls: np.array_split([i for i, label in enumerate(y_train) if label == cls], num_users) for cls in range(nclasses)}
    
    #print(f'IID Spliting: {ind2label}')
    for j in range(num_users):
        for cls in range(nclasses):
            partitions_train[j] = np.hstack([partitions_train[j], ind2label[cls][j]])
        
        dataidx = partitions_train[j]
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        partitions_train_stat[j] = tmp
        
        partitions_test[j] = np.hstack([partitions_test[j], idxs_test])

        dataidx = partitions_test[j]
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        partitions_test_stat[j] = tmp
        
    for j in range(num_users):
        partitions_train[j] = np.array(train_inds)[partitions_train[j]]
        
    return (partitions_train, partitions_test, partitions_train_stat, partitions_test_stat)

def get_partitions(num_users, train_ds_global, test_ds_global, args):
    
    if args.dataset == 'cifar10' and args.clustering_setting == '3_clusters' and not args.old_type:
        nclasses = 10
        
        X_train = train_ds_global.data
        Y_train = np.array(train_ds_global.target)

        X_test = test_ds_global.data
        Y_test = np.array(test_ds_global.target)
        
        indices = list(range(len(train_ds_global)))
        ind2label = {cls: [i for i, label in enumerate(Y_train) if label == cls] for cls in range(nclasses)}
        random.shuffle(indices)

        partitions_train = []
        partitions_test = []
        partitions_train_stat = []
        partitions_test_stat = []
        
        for k in range(len(num_users)):
            if k == 0:
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    r_inds = np.random.choice(np.array(ind2label[cls]), 500, replace=False)
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                    
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
            elif k == 1:
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    r_inds = np.random.choice(np.array(ind2label[cls]), 2000, replace=False)
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
            else: 
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    #r_inds = np.random.choice(np.array(ind2label[cls]), 2000, replace=False)
                    r_inds = np.array(ind2label[cls])
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
                
            Y_train_temp = Y_train[inds_subset]

            if args.partition == 'niid-labeldir':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= dir_partition(num_users[k], niid_beta=args.niid_beta, nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)
            elif args.partition == 'iid':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= iid_partition(num_users[k], nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)

            partitions_train.append(partitions_train_tmp)
            partitions_test.append(partitions_test_tmp)
            partitions_train_stat.append(partitions_train_stat_tmp)
            partitions_test_stat.append(partitions_test_stat_tmp)
        
    elif args.dataset == 'cifar10' and args.clustering_setting == '3_clusters' and args.old_type:
        print('!!!!!!!!!!!!!!!!!!!!! OLD TYPE PARTITIONING !!!!!!!!!!!!!!!!!!!!!!')
        nclasses = 10
        
        X_train = train_ds_global.data
        Y_train = np.array(train_ds_global.target)

        X_test = test_ds_global.data
        Y_test = np.array(test_ds_global.target)
        
        indices = list(range(len(train_ds_global)))
        random.shuffle(indices)

        partitions_train = []
        partitions_test = []
        partitions_train_stat = []
        partitions_test_stat = []
        
        for k in range(len(num_users)):
            if k == 0:
                inds_subset = indices[0:5000]
            elif k == 1:
                inds_subset = indices[5000:25000]
            else: 
                inds_subset = indices[25000:]
                
            Y_train_temp = Y_train[inds_subset]

            if args.partition == 'niid-labeldir':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= dir_partition(num_users[k], niid_beta=args.niid_beta, nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)
            elif args.partition == 'iid':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= iid_partition(num_users[k], nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)

            partitions_train.append(partitions_train_tmp)
            partitions_test.append(partitions_test_tmp)
            partitions_train_stat.append(partitions_train_stat_tmp)
            partitions_test_stat.append(partitions_test_stat_tmp)
            
    elif args.dataset == 'cifar100' and args.clustering_setting == '3_clusters' and not args.old_type:
        print('CIFAR-100 Partitioning for 3 clusters setting')
        nclasses = 100
        
        X_train = train_ds_global.data
        Y_train = np.array(train_ds_global.target)

        X_test = test_ds_global.data
        Y_test = np.array(test_ds_global.target)
        
        indices = list(range(len(train_ds_global)))
        ind2label = {cls: [i for i, label in enumerate(Y_train) if label == cls] for cls in range(nclasses)}
        random.shuffle(indices)

        partitions_train = []
        partitions_test = []
        partitions_train_stat = []
        partitions_test_stat = []
        
        for k in range(len(num_users)):
            if k == 0:
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    r_inds = np.random.choice(np.array(ind2label[cls]), 50, replace=False)
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                    
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
            elif k == 1:
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    r_inds = np.random.choice(np.array(ind2label[cls]), 200, replace=False)
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
            else: 
                inds_subset = np.array([], dtype='int')
                for cls in range(nclasses):
                    #r_inds = np.random.choice(np.array(ind2label[cls]), 2000, replace=False)
                    r_inds = np.array(ind2label[cls])
                    inds_subset = np.hstack([inds_subset, r_inds])
                    ind2label[cls] = [i for i in ind2label[cls] if i not in r_inds]
                
                random.shuffle(inds_subset)
                inds_subset = list(inds_subset)
                
            Y_train_temp = Y_train[inds_subset]

            if args.partition == 'niid-labeldir':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= dir_partition(num_users[k], niid_beta=args.niid_beta, nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)
            elif args.partition == 'iid':
                partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
                partitions_test_stat_tmp= iid_partition(num_users[k], nclasses=nclasses, 
                                                        y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)

            partitions_train.append(partitions_train_tmp)
            partitions_test.append(partitions_test_tmp)
            partitions_train_stat.append(partitions_train_stat_tmp)
            partitions_test_stat.append(partitions_test_stat_tmp)
        
        
    return partitions_train, partitions_test, partitions_train_stat, partitions_test_stat