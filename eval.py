import bottleneck as bn
import numpy as np
import pandas as pd
import os
from metrics import precision_at_k_batch, recall_at_k_batch, NDCG_at_k_batch
import time

from embedding import PersonalizedEmbedding as pe

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from joblib import Parallel, delayed

import logging
log = logging.getLogger("rec_eval")

def _predict(user, item, context_items):
    '''
    predict user's preference to an item
    user: target user id
    item: target item id
    context_items: ids of previous items of the user
    '''
    pred = pe.predict(user, item, context_items)
    return 0.5


def prediction_batch(train_data, theta, beta, rho,
                     user_idx, mu=None, vad_data=None):
    """
    Input:
    ======
    X (scipy.sparse.csr_matrix) : (n_users, n_factors) matrix
    Y (scipy.sparse.csr_matrix) : (n_items, n_factors) matrix
    train_data (scipy.sparse.csr_matrix) : (n_users, n_items) matrix

    Return:
    =======
    prediction score matrix (numpy.array) : (n_users, n_items) matrix
    """
    n_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    item_idx = np.zeros((n_users, n_items), dtype=bool)

    # exclude examples from training and validation (if any)
    item_idx[train_data[user_idx].nonzero()] = True
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = True

    X_pred = X[user_idx].dot(Y)

    if mu is not None:
        if isinstance(mu, np.ndarray):
            assert mu.size == n_items  # mu_i
            X_pred *= mu
        elif isinstance(mu, dict):  # func(mu_ui)
            params, func = mu['params'], mu['func']
            args = [params[0][user_idx], params[1]]
            if len(params) > 2:  # for bias term in document or length-scale
                args += [params[2][user_idx]]
            if not callable(func):
                raise TypeError("expecting a callable function")
            X_pred *= func(*args)
        else:
            raise ValueError("unsupported mu type")
    # remove items in training and validation set from the
    # prediction result
    X_pred[item_idx] = -np.inf
    return X_pred