import torch
import numpy as np
import pandas as pd
import time
from models import GRAPH2TAXO
from global_config import Config, Backends
from scipy.sparse import coo_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import logging
import torch.nn.functional as F
from utils import F1_Loss, load_embeddings, logger_init

class graph2taxoSupervisor:
    def __init__(self):
    
        # Logger
        self.logger = logger_init()
        # Use Cuda
        Config.cuda = True
        self.device = None
        if Config.cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        ################## Data ###################
        # Load Sparse Adjacency Matrix
        file_name = 'adj_input.pkl'
        (data, rows, columns, vocab_dict) = pd.read_pickle(file_name)
        id_word_map = {v: k for k, v in vocab_dict.items()}
        rel_list = ['ISA']
        num_entities = len(vocab_dict)
        num_relations = len(rel_list)
    
        # Build the adjacency matrix and remove the edges which fre < 10.
        rows = rows + [i for i in range(num_entities)]
        columns = columns + [i for i in range(num_entities)]
        data = data + [1 for i in range(num_entities)]
        adjs = coo_matrix((data, (rows, columns)), shape=(num_entities, num_entities)).toarray()
        adjs = np.where(adjs >= 10, 1, 0)
        self.adjs = torch.FloatTensor(adjs).to(device=self.device)
        del rows
        del columns
        del data

        # Use X as index for the randomly initialized embeddings
        self.X = torch.LongTensor([i for i in range(num_entities)]).to(device=self.device)
        # Load the word embedding if we use it.
        self.word_embs = load_embeddings(vocab_dict).to(device=self.device)
        logging.info('Finished the preprocessing')
    
        ################## Model, Optimizer, LossFunction ###################
        self.model = GRAPH2TAXO(num_entities, num_relations).to(device=self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
        self.f1_loss = F1_Loss().to(device=self.device)
    
        ################## Part of Hyperparameters ###################
        # Hyperparameters for the constraints
        self.lambda_A = 1.0  # 1.0
        self.c_A = 0.5  # 0.5
        self.tau_A = 1.0  # 1.0

    def _matrix_poly(self, matrix, d):
        x = torch.eye(d).to(device=self.device) + torch.div(matrix, d)
        return torch.matrix_power(x, d)

    def _h_A(self, A, m):
        expm_A = self._matrix_poly(torch.mul(A, A), m)
        h_A = (torch.trace(expm_A) - m)
        return h_A

    def _connectivity(self, A, m, i, labels):
        expm_A = self._matrix_poly(torch.mul(A, A), m)
        connectivity = torch.sigmoid(torch.sum(expm_A[:, labels[i][11]], dim=1))
        return connectivity

    def _run_one(self, labels, i, loss_arr, pre_arr, rec_arr, fs_arr, ave_pre, fs_max, fs_max_taxo, state):
        if state == 'train':
            self.model.train()
            self.opt.zero_grad()
        key = labels[i][0]
        terms = torch.LongTensor(key).to(device=self.device)
        e1 = torch.LongTensor(labels[i][1]).to(device=self.device)
        e2 = torch.LongTensor(labels[i][2]).to(device=self.device)
        rel = torch.LongTensor(labels[i][3]).to(device=self.device)
        label = torch.FloatTensor(labels[i][4]).to(device=self.device)
        taxo = labels[i][5]
        e1_index = torch.LongTensor(labels[i][6]).to(device=self.device)
        e2_index = torch.LongTensor(labels[i][7]).to(device=self.device)
        fre = torch.FloatTensor(labels[i][8]).to(device=self.device)
        degree = torch.FloatTensor(labels[i][9]).to(device=self.device)
        substr = torch.FloatTensor(labels[i][10]).to(device=self.device)
        pred = self.model.forward(e1, e2, rel, self.X, self.adjs, terms, e1_index, e2_index, self.word_embs, fre,
                                  degree, substr)

        label = label.view(label.size()[0], 1)
        loss = self.f1_loss(pred, label)

        pred_DAG = F.relu(pred - 0.5) * 2

        indices = torch.LongTensor([labels[i][6], labels[i][7]]).to(device=self.device)
        pred_data = pred_DAG.view(pred_DAG.size()[0])
        A_pred = torch.sparse_coo_tensor(indices, pred_data, torch.Size([len(key), len(key)]),
                                         requires_grad=True).to_dense()
        h_A = self._h_A(A_pred, len(key)) / float(len(key))
        loss += self.lambda_A * h_A + 0.5 * self.c_A * h_A * h_A

        connectivity = self._connectivity(A_pred, len(key), i, labels)
        loss_con = self.model.loss(connectivity, torch.FloatTensor([1 for i in range(connectivity.size()[0])]).to(
            device=self.device))
        loss += self.tau_A * loss_con
        if state == 'train':
            loss.backward()
            self.opt.step()

        num_keys = len(key)
        key_index = {}
        count = 0
        for n_k in range(len(key)):
            key_index[key[n_k]] = count
            count += 1
        pred = pred.cpu().detach().numpy().flatten()
        head = e1.cpu().detach().numpy().flatten()
        tail = e2.cpu().detach().numpy().flatten()
        pred_m = [[0 for x in range(num_keys)] for y in range(num_keys)]
        for num in range(len(head)):
            x = key_index[head[num]]
            y = key_index[tail[num]]
            pred_m[x][y] = pred[num]

        pred_m = np.array(pred_m).flatten()
        taxo = np.array(taxo).flatten()
        precision_curve, recall_curve, thr_curve = precision_recall_curve(taxo, pred_m)
        f_max_taxo = 0
        Pre = 0
        Rec = 0
        threshold = 0
        for n_t in range(len(precision_curve)):
            if (precision_curve[n_t] + recall_curve[n_t]) != 0:
                F1 = 2 * (precision_curve[n_t] * recall_curve[n_t]) / (precision_curve[n_t] + recall_curve[n_t])
                if F1 > f_max_taxo:
                    f_max_taxo = F1
                    Pre = precision_curve[n_t]
                    Rec = recall_curve[n_t]
                    threshold = thr_curve[n_t]

        y_true = label.cpu().detach().numpy().flatten()
        y_pred = pred

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
        f_max = 0
        for n_t in range(len(precision_curve)):
            if (precision_curve[n_t] + recall_curve[n_t]) != 0:
                F1 = 2 * (precision_curve[n_t] * recall_curve[n_t]) / (precision_curve[n_t] + recall_curve[n_t])
                if F1 > f_max:
                    f_max = F1

        average_p = average_precision_score(y_true, y_pred)
        value = 0.5
        y_pred_t = np.where(pred_m > value, 1, 0)
        p, r, f, _ = precision_recall_fscore_support(taxo, y_pred_t, warn_for=tuple())

        loss_arr.append(loss.item())
        pre_arr.append(p[1])
        rec_arr.append(r[1])
        fs_arr.append(f[1])
        ave_pre.append(average_p)
        fs_max.append(f_max)
        fs_max_taxo.append(f_max_taxo)

        if state == 'sep_semeval':
            self.logger.info(
                'Nodes: {:d} Taxo: loss= {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, BestF_Precision = {:.4f}, BestF_Recall= {:.4f}, Best_Threshold = {:.4f}, Average_precision = {:.4f}, Max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                    .format(len(labels[i][0]), loss.item(), p[1], r[1], f[1], Pre, Rec, threshold, average_p, f_max, f_max_taxo))

        return loss_arr, pre_arr, rec_arr, fs_arr, ave_pre, fs_max, fs_max_taxo

    ################## Train ##################
    def train(self, epoch, labels, state):
        t = time.time()
        loss_arr = []
        pre_arr = []
        rec_arr = []
        fs_arr = []
        ave_pre = []
        fs_max = []
        fs_max_taxo = []

        for i in range(len(labels)):
            loss_arr, pre_arr, rec_arr, fs_arr, ave_pre, fs_max, fs_max_taxo = \
                self._run_one(labels, i, loss_arr, pre_arr, rec_arr, fs_arr, ave_pre, fs_max, fs_max_taxo, state)

        ave_loss = sum(loss_arr) / float(len(loss_arr))
        precision = sum(pre_arr) / float(len(pre_arr))
        recall = sum(rec_arr) / float(len(rec_arr))
        fscore = sum(fs_arr) / float(len(fs_arr))
        average_precision = sum(ave_pre) / float(len(ave_pre))
        fscore_max = sum(fs_max) / float(len(fs_max))
        fscore_max_taxo = sum(fs_max_taxo) / float(len(fs_max_taxo))

        self.logger.info('Epoch: {:04d}'.format(epoch + 1))
        self.logger.info(
            'Train Results: loss= {:.4f}, time: {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, Average_precision = {:.4f}, Average_max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                .format(ave_loss, time.time() - t, precision, recall, fscore, average_precision, fscore_max,
                        fscore_max_taxo))


    ################## TEST ##################
    def test(self, epoch, labels, state):
        t_t = time.time()
        self.model.eval()
        with torch.no_grad():
            loss_arr = []
            pre_arr = []
            rec_arr = []
            fs_arr = []
            ave_pre = []
            fs_max = []
            fs_max_taxo = []

            for i in range(len(labels)):
                loss_arr, pre_arr, rec_arr, fs_arr, ave_pre, fs_max, fs_max_taxo = \
                    self._run_one(labels, i, loss_arr, pre_arr, rec_arr, fs_arr, ave_pre, fs_max, fs_max_taxo, state)

            ave_loss = sum(loss_arr) / float(len(loss_arr))
            precision = sum(pre_arr) / float(len(pre_arr))
            recall = sum(rec_arr) / float(len(rec_arr))
            fscore = sum(fs_arr) / float(len(fs_arr))
            average_precision = sum(ave_pre) / float(len(ave_pre))
            fscore_max = sum(fs_max) / float(len(fs_max))
            fscore_max_taxo = sum(fs_max_taxo) / float(len(fs_max_taxo))

            if state != 'sep_semeval':
                self.logger.info(
                    '{:s} Results: loss= {:.4f}, time: {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, Average_precision = {:.4f}, Average_max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                        .format(state, ave_loss, time.time() - t_t, precision, recall, fscore, average_precision, fscore_max,
                                fscore_max_taxo))
        return fscore






