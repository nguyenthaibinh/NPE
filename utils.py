import numpy as np


def _minibatch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def _gpu(tensor, gpu=False):
    if gpu:
        return tensor.cuda()
    else:
        return tensor


def _cpu(tensor):
    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def _shuffle(users, items):
    shuffle_indices = np.arange(len(users))
    np.random.shuffle(shuffle_indices)

    return (users[shuffle_indices].astype(np.int64),
            items[shuffle_indices].astype(np.int64))
