import numpy as np
import pandas as pd
import argparse

TRAIN_FILE = '/Users/binhnguyen/work/datasets/ml_1m/implicit_data/per_user/train.csv'

def toy_data(train_file):
    users = []
    items = []
    contexts = []
    labels = []

    with open(train_file) as f:
        temp = f.read().splitlines()
    n = len(temp)
    for i in range(n):
        s = temp[i].split()
        user_id = int(s[0])
        item_id = int(s[1])
        context_ids = np.array(s[2].split(','), dtype=int)
        users.append(user_id)
        items.append(item_id)
        contexts.append(context_ids)
        labels.append(int(s[3]))
    n_users = max(users) + 1
    n_items = max(items) + 1
    dat = [np.array(users), np.array(items),
           np.array(contexts), np.array(labels)]
    return dat, n_users, n_items


def data_standardize(dat, n_users, n_items):
    users, items, contexts, labels = dat[0], dat[1], dat[2], dat[3]
    new_contexts = []
    new_users = []
    new_items = []
    n_len = len(users)
    for i in range(n_len):
        user_vec = np.zeros(n_users, dtype=float)
        user_vec[users[i]] = 1
        new_users.append(user_vec)

        item_vec = np.zeros(n_items, dtype=float)
        item_vec[items[i]] = 1
        new_items.append(item_vec)

        ctx_vec = np.zeros(n_items, dtype=float)
        # print ("contexts[i]:", contexts[i])
        ctx_vec[contexts[i]] = 1
        # print ("ctx_vec:", ctx_vec)
        new_contexts.append(ctx_vec)

    users = np.array(new_users, dtype=float)
    items = np.array(new_items, dtype=float)
    contexts = np.array(new_contexts, dtype=float)
    return [users,
            items,
            contexts, labels]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-file', default="train.csv", metavar='O',
                    help='train file')

    args = parser.parse_args()

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

    n_dat = len(users)
    print(n_dat)
    contexts = []
    for i in range(n_dat):
        uid = users[i]
        iid = items[i]
        item_list = hist[uid]
        tmp_context = np.zeros(n_items)
        tmp_context[item_list] = 1
        tmp_context[iid] = 0
        contexts.append(tmp_context)
    context_mat = np.array(contexts, dtype=np.float32)
    np.savez("train.npz", users=users, items=items,
             contexts=context_mat)
                # print("contexts:", contexts)
