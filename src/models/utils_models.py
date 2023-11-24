from torch import nn
import torch.nn.init as init

import torchvision
import timm
import copy
from . import *

def get_models_fedmh(num_users, model, dataset, args):

    users_model = []

    for i in range(-1, num_users):
        if model == "resnet8":
            if dataset in ("cifar10"):
                net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=100)
        elif model == "resnet14-0.75":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [1,1,2,2], scaling=0.75, num_classes=10)
        elif model == "resnet14":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [1,2,2,1], scaling=1.0, num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet(BasicBlock, [1,2,2,1], scaling=1.0, num_classes=100)
        elif model == "resnet18":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [2,2,2,2], scaling=1.0, num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet(BasicBlock, [2,2,2,2], scaling=1.0, num_classes=100)
        elif model == "resnet34":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [3,4,6,3], num_classes=10)
            elif dataset in ("cifar100"):
                net = ResNet(BasicBlock, [3,4,6,3], num_classes=100)
        elif model == "vgg7":
            if dataset in ("cifar10"):
                net = vgg7_bn(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg7_bn(num_classes=100)
        elif model == "vgg12":
            if dataset in ("cifar10"):
                net = vgg12_bn(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg12_bn(num_classes=100)
        elif model == "vgg11":
            if dataset in ("cifar10"):
                net = vgg11_bn(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg11_bn(num_classes=100)
        elif model == "vgg16":
            if dataset in ("cifar10"):
                net = vgg16_bn(num_classes=10)
            elif dataset in ("cifar100"):
                net = vgg16_bn(num_classes=100)
        elif model == "squeezenet1_0":
            if dataset in ("cifar10"):
                net = torchvision.models.squeezenet1_0(pretrained=False, num_classes=10)
        elif model == "regnetx_002":
            if dataset in ("cifar10"):
                net = timm.create_model('regnetx_002', pretrained=False, num_classes=10)
        elif model == "shufflenet_v2_x1_0":
            if dataset in ("cifar10"):
                net = torchvision.models.shufflenet_v2_x1_0(pretrained=False, num_classes=10)
        elif model == "densenet121":
            if dataset in ("cifar10"):
                net = torchvision.models.densenet121(pretrained=False, num_classes=10)
        elif model == "efficientnet-b3":
            if dataset in ("cifar10"):
                net = torchvision.models.efficientnet_b3(pretrained=False, num_classes=10)
        elif model == "mobilenet_v2":
            if dataset in ("cifar10"):
                net = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)
        elif model == "edgenext_xx_small":
            if dataset in ("cifar10"):
                net = timm.create_model('edgenext_xx_small', pretrained=False, num_classes=10)
        elif model == "edgenext_x_small":
            if dataset in ("cifar10"):
                net = timm.create_model('edgenext_x_small', pretrained=False, num_classes=10)
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


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    
    return 