#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
from utils.sampling import mnist_noniid
from utils.options import args_parser
from models.Update import LocalUpdate, test_inference
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.utils import exp_details, powersettool, shapley
from utils.sv import exact


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # Powerset list
    powerset = list(powersettool(range(1, args.num_users + 1)))  # generate a powerset list of tuples

    # Initialize all the sub-models
    submodel_dict = {}
    for subset in powerset[:-1]:  # exclude only the global set. the null set still has a random initialized model
        submodel_dict[subset] = copy.deepcopy(net_glob)
        submodel_dict[subset].to(args.device)
        submodel_dict[subset].train()

    # dictionary containing all the accuracies for the subsets
    accuracy_dict = {}

    ######## Timing starts ########
    start_time = time.time() # start the timer

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in tqdm(range(args.epochs)):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(1, args.num_users+1), m, replace=False)
        total_data = sum(len(dict_users[i]) for i in range(1, m + 1))
        fraction = [len(dict_users[i]) / total_data for i in range(1, m + 1)]
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals, fraction)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if (iter+1) % print_every == 0:
            print(f' \nAvg Training Stats after {iter+1} global rounds:')
            # print(f'Training Loss : {train_loss}') # print check
            print(f'Training Loss : {np.mean(np.array(loss_train))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        test_acc, test_loss = test_inference(args, net_glob, dataset_test)

        print(f' \n Results after {args.epochs} global rounds of training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

        if args.sv == 'exact':
            exact(args, net_glob, powerset, submodel_dict, fraction, dataset_train, dataset_test, dict_users, test_acc)



    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

