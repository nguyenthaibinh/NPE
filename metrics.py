import numpy as np
'''
precision
recall
mrr
nDCG
'''

def precision_at_k_batch(X_pred_binary, X_true_binary, k=10, normalize=True):
    """
    Calculate precision@k

    Input:
    ======
    X_pred_binary (scipy.sparse.csr_matrix) : prediction matrix (binary data)
    X_pred_binary (scipy.sparse.csr_matrix) : true data (binary data)
    k (int) : number of items to be recommended

    Output:
    =======
    precision (float)
    """
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    if normalize:
        precision = tmp / np.minimum(k, X_pred_binary.sum(axis=1))
    else:
        precision = tmp / k

    return precision


def recall_at_k_batch(X_pred_binary, X_true_binary, k=10):
    """
    Calculate recall@k

    Input:
    ======
    X_pred_binary (scipy.sparse.csr_matrix) : prediction matrix (binary data)
    X_pred_binary (scipy.sparse.csr_matrix) : true data (binary data)
    k (int) : number of items to be recommended

    Output:
    =======
    precision (float)
    """
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def NDCG_at_k_batch(X_pred, X_true, k):
    batch_users = X_pred.shape[0]

    # Get item_id of top-k items
    idx_topk_part = bn.argpartsort(-X_pred, k, axis=1)
    topk_items = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_items, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (X_true[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in X_true.getnnz(axis=1)])
    return DCG / IDCG