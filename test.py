from __future__ import print_function
from __future__ import print_function
import argparse
from embeddings import *
import pandas as pd
import torch
import numpy as np
from utils import _cpu

# TRAIN_FILE = '/Users/binhnguyen/work/datasets/ml_1m/implicit_data/per_user/train.csv'


if __name__ == "__main__":
    factors = 10
    epochs = 5
    batch_size = 4
    learning_rate = 0.001

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for training (default: 10000)')
    parser.add_argument('--factors', type=int, default=5, metavar='N',
                    help='number of factors (default: 10)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
    parser.add_argument('--verbose', type=int, default=0, metavar='verbose',
                    help='learning rate (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--optimizer', default="sgd", metavar='O',
                    help='random seed (default: SGD)')
    parser.add_argument('--model', default="emb-cf", metavar='O',
                    help='random seed (default: EMB-CF)')
    parser.add_argument('--train-file', default="train.csv", metavar='O',
                    help='train file')
    parser.add_argument('--is-test', type=int, default="0", metavar='O',
                    help='is test (default 0')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    '''
    dat, n_users, n_items = data.toy_data(train_file='train.txt')
    n_users = 500000
    n_items = 400000
    dat = data.data_standardize(dat, n_users, n_items)
    users = dat[0]
    '''
    if args.is_test:
        n_users = 3
        n_items = 9
        users = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        items = np.array([0, 1, 2, 3, 1, 2, 6, 5, 2, 6, 7, 4, 8])
        hist = [[0, 1, 2], [3, 1, 2, 6, 5], [2, 6, 7, 4, 8]]
    else:
        try:
            TRAIN_FILE = args.train_file
            train_raw = pd.read_csv(TRAIN_FILE, sep=";")
            n_users = train_raw['uid'].max() + 1
            n_items = train_raw['iid'].max() + 1
            # print("n_users: {}, n_items: {}".format(n_users, n_items))
            train = train_raw.groupby('uid')['iid'].apply(list).reset_index()
            train = train.sort_values(by=['uid'])
            # print("train:", train)
            users = np.array(train_raw['uid'])
            items = np.array(train_raw['iid'])
            hist = train['iid']
            # print("hist:", hist)
        except Exception as e:
            print("Cannot read file")

    dat = [users, items, hist]

    # data = {0: [0, 1, 2, 1, 4], 1: [3, 1, 2, 6], 2: [2, 6, 7, 8]}

    model = EmbeddingModel(n_factors=args.factors, n_epochs=args.epochs,
                           batch_size=args.batch_size,
                           learning_rate=args.lr,
                           use_cuda=args.cuda, optimizer=args.optimizer,
                           model=args.model)
    emb = model.fit(dat, n_users, n_items, verbose=args.verbose)

    model_file = "{}_vectors.npz".format(args.model)

    if args.model == "mf":
        theta, beta = emb.get_embeddings()
        theta = _cpu(theta.weight.data).numpy()
        beta = _cpu(beta.weight.data).numpy()

        np.savez(model_file, theta=theta, beta=beta)
    elif args.model == "emb":
        beta, rho = emb.get_embeddings()
        beta = _cpu(beta.weight.data).numpy()
        rho = _cpu(rho.weight.data).numpy()

        np.savez(model_file, beta=beta, rho=rho)
    else:
        theta, beta, rho = emb.get_embeddings()
        theta = _cpu(theta.weight.data).numpy()
        beta = _cpu(beta.weight.data).numpy()
        rho = _cpu(rho.weight.data).numpy()

        np.savez(model_file, theta=theta, beta=beta, rho=rho)

    '''
    load = np.load("embedding_vectors.npz")
    print(load['theta'], type(load['theta']))
    '''
