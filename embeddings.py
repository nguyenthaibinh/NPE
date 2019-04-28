import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import utils
import time
from utils import _gpu, _cpu, _minibatch, _shuffle


class DualEmbedding(nn.Module):
    def __init__(self, n_users, n_items, n_factors=40, dropout_p=0, sparse=False):
        super(DualEmbedding, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        self.user_embeddings = nn.Linear(int(n_users), n_factors)
        self.item_embeddings = nn.Linear(int(n_items), n_factors)
        self.item_contexts = nn.Linear(int(n_items), n_factors)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.sparse = sparse

        self.sigmoid = nn.Sigmoid()

    def forward(self, user_vec, item_vec, context_vec):
        # print("context_vec:", context_vec)
        theta_u = self.user_embeddings(user_vec)
        beta_i = self.item_embeddings(item_vec)
        rho_ui = self.item_contexts(context_vec)

        user_item_prod = (theta_u * beta_i).sum(1)
        item_context_prod = (beta_i * rho_ui).sum(1)
        preds = self.sigmoid(user_item_prod + item_context_prod)
        return preds

    def predict(self, user, item, contexts):
        return self.forward(user, item, contexts)


# Model
class PersonalizedEmbedding(nn.Module):
    def __init__(self, n_users, n_items, n_factors=40,
                 dropout_p=0, sparse=False, user_history=None):
        super(PersonalizedEmbedding, self).__init__()
        self._hist = user_history
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        # user feature vectors
        self.theta = nn.Embedding(int(n_users), n_factors, sparse=sparse)
        # item embedding vectors
        self.beta = nn.Embedding(int(n_items), n_factors, sparse=sparse)
        # item context vectors
        self.rho = nn.Embedding(int(n_items), n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.sparse = sparse

        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item, contexts):
        # users, items, contexts = data[0], data[1], data[2]
        theta_u = self.theta(user)
        beta_i = self.beta(item)
        rho_ui = self.rho(contexts).sum(dim=1)
        rho_ui = torch.squeeze(rho_ui)

        user_item_prod = (theta_u * beta_i).sum(1)
        item_context_prod = (beta_i * rho_ui).sum(1)
        preds = self.sigmoid(user_item_prod + item_context_prod)
        return preds

    def get_embeddings(self):
        return self.theta, self.beta, self.rho

    def user_rep(self, user_id):
        user_var = Variable(torch.from_numpy(np.array([user_id])))
        u_emb = self.theta(user_var)
        return u_emb

    def item_rep(self, item_id):
        item_var = Variable(torch.from_numpy(np.array([item_id])))
        i_emb = self.beta(item_var)
        i_ctx = self.rho(item_var)
        return i_emb, i_ctx

    def predict(self, user, item, contexts):
        return self.forward(user, item, contexts)


# Model
class PerEmbedding(nn.Module):
    def __init__(self, n_users, n_items, n_factors=40,
                 dropout_p=0, sparse=False, user_history=None,
                 use_cuda=True):
        super(PerEmbedding, self).__init__()
        self._hist = user_history
        self._n_users = n_users
        self._n_items = n_items
        self._n_factors = n_factors
        self._use_cuda = use_cuda

        # user feature vectors
        self.theta = nn.Embedding(int(n_users), n_factors, sparse=sparse)
        # item embedding vectors
        self.beta = nn.Embedding(int(n_items), n_factors, sparse=sparse)
        # item context vectors
        self.rho = nn.Embedding(int(n_items), n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.sparse = sparse

        self.sigmoid = nn.Sigmoid()

        item_var = np.array(range(n_items))
        item_var = torch.from_numpy(item_var)
        self._item_var = Variable(_gpu(item_var, self._use_cuda))

    def forward(self, users, items, contexts):
        # users, items, contexts = data[0], data[1], data[2]
        theta_u = self.theta(users)
        beta_i = self.beta(items)
        user_item_prod = (theta_u * beta_i).sum(1)
        preds = self.sigmoid(user_item_prod)
        return preds

    def get_embeddings(self):
        return self.theta, self.beta, self.rho

    def user_rep(self, user_id):
        user_var = Variable(torch.from_numpy(np.array([user_id])))
        u_emb = self.theta(user_var)
        return u_emb

    def item_rep(self, item_id):
        item_var = Variable(torch.from_numpy(np.array([item_id])))
        i_emb = self.beta(item_var)
        i_ctx = self.rho(item_var)
        return i_emb, i_ctx

    def predict(self, user, item, contexts):
        return self.forward(user, item, contexts)


class EmbeddingModel(object):
    def __init__(self, loss='pointwise', n_factors=50, n_epochs=100,
                 batch_size=100, use_cuda=True, sparse=False, model="MF",
                 optimizer='adam', l2=0.0, learning_rate=0.001):
        self._model = model
        self._n_factors = n_factors
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._l2 = l2
        self._lr = learning_rate
        self._loss = loss
        self._optimizer = optimizer

    def fit(self, dat, n_users, n_items, verbose=False):
        # users, items = dat[0], dat[1]
        # n_users = max(users) + 1
        # n_items = max(items) + 1

        users = dat[0]
        items = dat[1]
        hist = dat[2]
        cur_loss = None

        model_name = self._model.lower()

        if (model_name == "mf"):
            print("model=matrix factorization")
            model = BinaryMF(n_users=n_users, n_items=n_items,
                             n_factors=self._n_factors, dropout_p=0)

        else:
            print("model=EMB-CF")
            model = PerEmbedding(n_users=n_users, n_items=n_items,
                                 n_factors=self._n_factors, dropout_p=0,
                                 user_history=hist,
                                 use_cuda=self._use_cuda)
        model = _gpu(model, self._use_cuda)

        if self._loss == 'pointwise':
            loss_func = nn.BCELoss(weight=None, size_average=False)
        else:
            loss_func = nn.BCELoss(weight=None, size_average=False)

        if self._optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), weight_decay=self._l2)
        elif self._optimizer == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), weight_decay=self._l2)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self._lr)

        # epoch_loss = 0.0
        for epoch in range(self._n_epochs):
            # users, items, contexts, labels = utils._shuffle(dat)
            shuffled_users, shuffled_items = _shuffle(users, items)

            user_ids_tensor = _gpu(torch.from_numpy(shuffled_users),
                                   self._use_cuda)
            item_ids_tensor = _gpu(torch.from_numpy(shuffled_items),
                                   self._use_cuda)

            tmp_loss = 0
            start_t = time.time()

            for (batch_user,
                 batch_item) in zip(_minibatch(user_ids_tensor,
                                               self._batch_size),
                                    _minibatch(item_ids_tensor,
                                               self._batch_size)):
                n_batch = len(batch_user)

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                label_list = [1] * len(batch_user)
                label_var = Variable(_gpu(torch.FloatTensor(label_list),
                                          self._use_cuda))
                '''
                contexts = []
                for i in range(n_batch):
                    uid = batch_user[i]
                    iid = batch_item[i]
                    item_list = hist[uid]
                    tmp_context = np.zeros(n_items)
                    tmp_context[item_list] = 1
                    tmp_context[iid] = 0
                    contexts.append(tmp_context)
                # print("contexts:", contexts)
                contexts = torch.FloatTensor(np.array(contexts, dtype=np.float32))
                context_var = Variable(_gpu(contexts, self._use_cuda),
                                       requires_grad=False)
                '''
                context_var = None

                preds = model(user_var, item_var, context_var)
                loss = loss_func(preds, label_var)
                tmp_loss += loss.data[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            cur_loss = tmp_loss / n_users
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_t))
            if verbose:
                print("Epoch: {}: loss: {}, {}".format(epoch, cur_loss, elapsed_time))
        return model

    def predict(self, user_ids, item_ids, context_ids):
        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(utils._gpu(user_ids, self._use_cuda))
        item_var = Variable(utils._gpu(item_ids, self._use_cuda))
        context_var = Variable(utils._gpu(context_ids, self._use_cuda))
        preds = self._model(user_var, item_var, context_var)

        return utils._cpu(preds.data).numpy().flatten()


class MatrixFactorization(object):
    def __init__(self, loss='pointwise', n_factors=50, n_epochs=100,
                 batch_size=100, use_cuda=True, sparse=False, model="MF",
                 optimizer='adam', l2=0.0, learning_rate=0.001):
        self._model = model
        self._n_factors = n_factors
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._l2 = l2
        self._lr = learning_rate
        self._loss = loss
        self._optimizer = optimizer

    def _shuffle(self, users, items):
        shuffle_indices = np.arange(len(users))
        np.random.shuffle(shuffle_indices)

        return (users[shuffle_indices].astype(np.int64),
                items[shuffle_indices].astype(np.int64))

    def fit(self, dat, n_users, n_items, verbose=False):
        # users, items = dat[0], dat[1]
        # n_users = max(users) + 1
        # n_items = max(items) + 1

        users = dat[0]
        items = dat[1]
        cur_loss = None

        # model_name = self._model.lower()

        print("model=matrix factorization")
        model = BinaryMF(n_users=n_users, n_items=n_items,
                         n_factors=self._n_factors, dropout_p=0)
        model = _gpu(model, self._use_cuda)

        if self._loss == 'pointwise':
            loss_func = nn.BCELoss(weight=None, size_average=False)
        else:
            loss_func = nn.BCELoss(weight=None, size_average=False)

        if self._optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), weight_decay=self._l2)
        elif self._optimizer == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), weight_decay=self._l2)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self._lr)

        # epoch_loss = 0.0
        for epoch in range(self._n_epochs):
            shuffled_users, shuffled_items = self._shuffle(users, items)

            user_ids_tensor = _gpu(torch.from_numpy(shuffled_users),
                                   self._use_cuda)
            item_ids_tensor = _gpu(torch.from_numpy(shuffled_items),
                                   self._use_cuda)

            tmp_loss = 0
            start_t = time.time()
            for (batch_user,
                 batch_item) in zip(_minibatch(user_ids_tensor,
                                               self._batch_size),
                                    _minibatch(item_ids_tensor,
                                               self._batch_size)):
                # print("batch:", batch_user, batch_item)

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                label_list = [1] * len(batch_user)
                label_var = Variable(_gpu(torch.FloatTensor(label_list),
                                          self._use_cuda))

                # user_var = Variable(torch.from_numpy(np.array(user_list)))
                # item_var = Variable(torch.from_numpy(np.array(item_list)))

                preds = model(user_var, item_var)
                loss = loss_func(preds, label_var)
                tmp_loss += loss.data[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            cur_loss = tmp_loss / n_users
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_t))
            if verbose:
                print("Epoch: {}: loss: {}, {}".format(epoch, cur_loss, elapsed_time))
        return model

    def predict(self, user_ids, item_ids, context_ids):
        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(utils._gpu(user_ids, self._use_cuda))
        item_var = Variable(utils._gpu(item_ids, self._use_cuda))
        context_var = Variable(utils._gpu(context_ids, self._use_cuda))
        preds = self._model(user_var, item_var, context_var)

        return utils._cpu(preds.data).numpy().flatten()
