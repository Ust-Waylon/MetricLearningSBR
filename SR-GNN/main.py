#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
import pandas as pd
from pathlib import Path
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='dataset name: tmall/retailrocket/lastfm')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=2, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--logit_type', type=str, default='dot', help='dot product or euclidean')
parser.add_argument('--scale', type=float, default=1.0, help='scale factor')
opt = parser.parse_args()
print(opt)


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None).squeeze('columns')
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions


def process_sessions(sessions):
    inputs, targets = [], []
    for session in sessions:
        for i in range(1, len(session)):
            inputs.append(session[:i])
            targets.append(session[i])
    return inputs, targets


def main():
    root = Path(__file__).resolve().parent.parent
    dataset_path = root / 'datasets' / opt.dataset
    if not dataset_path.exists():
        dataset_path = Path('../datasets') / opt.dataset
    if not dataset_path.exists() and Path(opt.dataset).exists():
        dataset_path = Path(opt.dataset)

    if (dataset_path / 'num_items.txt').exists():
        print(f"Loading data from {dataset_path} using new format")
        train_sessions = read_sessions(dataset_path / 'train.txt')
        train_data = process_sessions(train_sessions)
        
        valid_data = None
        if (dataset_path / 'valid.txt').exists():
            valid_sessions = read_sessions(dataset_path / 'valid.txt')
            valid_data = process_sessions(valid_sessions)
            
        test_sessions = read_sessions(dataset_path / 'test.txt')
        test_data_loaded = process_sessions(test_sessions)
        
        with open(dataset_path / 'num_items.txt', 'r') as f:
            n_node = int(f.readline())
            
        if opt.validation:
            if valid_data is None:
                train_data, valid_data = split_validation(train_data, opt.valid_portion)
            test_data = valid_data
        else:
            test_data = test_data_loaded
    else:
        train_data = pickle.load(open(dataset_path / 'train.txt', 'rb'))
        if opt.validation:
            train_data, valid_data = split_validation(train_data, opt.valid_portion)
            test_data = valid_data
        else:
            test_data = pickle.load(open(dataset_path / 'test.txt', 'rb'))
            
        if opt.dataset == 'diginetica':
            n_node = 43098
        elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
            n_node = 37484
        else:
            n_node = 310

    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
