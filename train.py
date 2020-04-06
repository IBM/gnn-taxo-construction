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

################## Logger and Cuda ###################
# Logger
logger = logger_init()
# Use Cuda
Config.cuda = True
Config.device = None
if Config.cuda and torch.cuda.is_available():
    Config.device = torch.device('cuda')
else:
    Config.device = torch.device('cpu')


##################   Load Data   ###################
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
adjs = torch.FloatTensor(adjs).to(device=Config.device)
del rows
del columns
del data

# Load labels
file_name = 'labels_input.pkl'
(train_labels, val_labels, semeval_labels, semeval_labels_RL, semeval_trial_labels) = pd.read_pickle(file_name)
semeval_trial_labels = semeval_trial_labels[1:]  # Trial Data

# Use X as index for the randomly initialized embeddings
X = torch.LongTensor([i for i in range(num_entities)]).to(device=Config.device)
# Load the word embedding if we use it.
word_embs = load_embeddings(vocab_dict)
logging.info('Finished the preprocessing')


################## Model, Optimizer, LossFunction ###################
model = GRAPH2TAXO(num_entities, num_relations).to(device=Config.device)
opt = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
f1_loss = F1_Loss().to(device=Config.device)

################## Part of Hyperparameters ###################
# Loss Options
loss_type = "F1"  # "F1" or "BCE"
# Num of epochs
epochs = 1000
# Hyperparameters for the constraints
lambda_A = 1.0 #1.0
c_A = 0.5 #0.5
tau_A = 1.0 #1.0

################## Train ##################
def train(epoch, labels):
    t = time.time()
    loss_arr = []
    pre_arr = []
    rec_arr = []
    fs_arr = []
    ave_pre = []
    fs_max = []
    fs_max_taxo = []

    for i in range(len(labels)):
        model.train()
        opt.zero_grad()
        key = labels[i][0]
        terms = torch.LongTensor(key).to(device=Config.device)
        e1 = torch.LongTensor(labels[i][1]).to(device=Config.device)
        e2 = torch.LongTensor(labels[i][2]).to(device=Config.device)
        rel = torch.LongTensor(labels[i][3]).to(device=Config.device)
        label = torch.FloatTensor(labels[i][4]).to(device=Config.device)
        taxo = labels[i][5]
        e1_index = torch.LongTensor(labels[i][6]).to(device=Config.device)
        e2_index = torch.LongTensor(labels[i][7]).to(device=Config.device)
        fre = torch.FloatTensor(labels[i][8]).to(device=Config.device)
        degree = torch.FloatTensor(labels[i][9]).to(device=Config.device)
        substr = torch.FloatTensor(labels[i][10]).to(device=Config.device)
        pred = model.forward(e1, e2, rel, X, adjs, terms, e1_index, e2_index, word_embs, fre, degree, substr)

        label = label.view(label.size()[0], 1)
        if loss_type == "F1":
            loss = f1_loss(pred, label)
        else:
            loss = model.loss(pred, label)

        # DAG constraint
        def matrix_poly(matrix, d):
            x = torch.eye(d).to(device=Config.device) + torch.div(matrix, d)
            return torch.matrix_power(x, d)

        def _h_A(A, m):
            expm_A = matrix_poly(torch.mul(A, A), m)
            h_A = (torch.trace(expm_A) - m)
            return h_A

        pred_DAG = F.relu(pred - 0.5) * 2

        indices = torch.LongTensor([labels[i][6], labels[i][7]]).to(device=Config.device)
        pred_data = pred_DAG.view(pred_DAG.size()[0])
        A_pred = torch.sparse_coo_tensor(indices, pred_data, torch.Size([len(key), len(key)]),
                                         requires_grad=True).to_dense()
        h_A = _h_A(A_pred, len(key)) / float(len(key))
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A

        def _connectivity(A, m):
            expm_A = matrix_poly(torch.mul(A, A), m)
            connectivity = torch.sigmoid(torch.sum(expm_A[:, labels[i][11]], dim=1))
            return connectivity

        connectivity = _connectivity(A_pred, len(key))
        loss_con = model.loss(connectivity, torch.FloatTensor([1 for i in range(connectivity.size()[0])]).to(
            device=Config.device))
        loss += tau_A * loss_con
        loss.backward()
        opt.step()

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
        precision_curve, recall_curve, _ = precision_recall_curve(taxo, pred_m)
        f_max_taxo = 0
        for n_t in range(len(precision_curve)):
            if (precision_curve[n_t] + recall_curve[n_t]) != 0:
                F1 = 2 * (precision_curve[n_t] * recall_curve[n_t]) / (precision_curve[n_t] + recall_curve[n_t])
                if F1 > f_max_taxo:
                    f_max_taxo = F1

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

    ave_loss = sum(loss_arr) / float(len(loss_arr))
    precision = sum(pre_arr) / float(len(pre_arr))
    recall = sum(rec_arr) / float(len(rec_arr))
    fscore = sum(fs_arr) / float(len(fs_arr))
    average_precision = sum(ave_pre) / float(len(ave_pre))
    fscore_max = sum(fs_max) / float(len(fs_max))
    fscore_max_taxo = sum(fs_max_taxo) / float(len(fs_max_taxo))

    logger.info('Epoch: {:04d}'.format(epoch + 1))
    logger.info(
        'Train Results: loss= {:.4f}, time: {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, Average_precision = {:.4f}, Average_max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
        .format(ave_loss, time.time() - t, precision, recall, fscore, average_precision, fscore_max, fscore_max_taxo))


################## TEST ##################
def test(epoch, labels, state):
    t_t = time.time()
    model.eval()
    with torch.no_grad():

        loss_arr = []
        pre_arr = []
        rec_arr = []
        fs_arr = []
        ave_pre = []
        fs_max = []
        fs_max_taxo = []

        for i in range(len(labels)):
            key = labels[i][0]
            terms = torch.LongTensor(key).to(device=Config.device)
            e1 = torch.LongTensor(labels[i][1]).to(device=Config.device)
            e2 = torch.LongTensor(labels[i][2]).to(device=Config.device)
            rel = torch.LongTensor(labels[i][3]).to(device=Config.device)
            label = torch.FloatTensor(labels[i][4]).to(device=Config.device)
            taxo = labels[i][5]
            e1_index = torch.LongTensor(labels[i][6]).to(device=Config.device)
            e2_index = torch.LongTensor(labels[i][7]).to(device=Config.device)
            fre = torch.FloatTensor(labels[i][8]).to(device=Config.device)
            degree = torch.FloatTensor(labels[i][9]).to(device=Config.device)
            substr = torch.FloatTensor(labels[i][10]).to(device=Config.device)
            pred = model.forward(e1, e2, rel, X, adjs, terms, e1_index, e2_index, word_embs, fre, degree, substr)

            label = label.view(label.size()[0], 1)
            if loss_type == "F1":
                loss = f1_loss(pred, label)
            else:
                loss = model.loss(pred, label)

            def matrix_poly(matrix, d):
                x = torch.eye(d).to(device=Config.device) + torch.div(matrix, d)
                return torch.matrix_power(x, d)

            def _h_A(A, m):
                expm_A = matrix_poly(torch.mul(A, A), m)
                h_A = torch.trace(expm_A) - m
                return h_A

            pred_DAG = F.relu(pred - 0.5) * 2

            indices = torch.LongTensor([labels[i][6], labels[i][7]]).to(device=Config.device)
            pred_data = pred_DAG.view(pred_DAG.size()[0])
            A_pred = torch.sparse_coo_tensor(indices, pred_data, torch.Size([len(key), len(key)]),
                                             requires_grad=True).to_dense()
            h_A = _h_A(A_pred, len(key)) / float(len(key))
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A

            def _connectivity(A, m):
                expm_A = matrix_poly(torch.mul(A, A), m)
                connectivity = torch.sigmoid(torch.sum(expm_A[:, labels[i][11]], dim=1))
                return connectivity

            connectivity = _connectivity(A_pred, len(key))
            loss_con = model.loss(connectivity, torch.FloatTensor([1 for i in range(connectivity.size()[0])]).to(
                device=Config.device))
            loss += tau_A * loss_con

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
            precision_curve, recall_curve, _ = precision_recall_curve(taxo, pred_m)
            f_max_taxo = 0
            for n_t in range(len(precision_curve)):
                if (precision_curve[n_t] + recall_curve[n_t]) != 0:
                    F1 = 2 * (precision_curve[n_t] * recall_curve[n_t]) / (precision_curve[n_t] + recall_curve[n_t])
                    if F1 > f_max_taxo:
                        f_max_taxo = F1

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

        ave_loss = sum(loss_arr) / float(len(loss_arr))
        precision = sum(pre_arr) / float(len(pre_arr))
        recall = sum(rec_arr) / float(len(rec_arr))
        fscore = sum(fs_arr) / float(len(fs_arr))
        average_precision = sum(ave_pre) / float(len(ave_pre))
        fscore_max = sum(fs_max) / float(len(fs_max))
        fscore_max_taxo = sum(fs_max_taxo) / float(len(fs_max_taxo))

        if state == 'val':
            logger.info(
                'Val Results: loss= {:.4f}, time: {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, Average_precision = {:.4f}, Average_max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                    .format(ave_loss, time.time() - t_t, precision, recall, fscore, average_precision, fscore_max,
                            fscore_max_taxo))
        elif state == 'test':
            logger.info(
                'Test Results: loss= {:.4f}, time: {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, Average_precision = {:.4f}, Average_max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                    .format(ave_loss, time.time() - t_t, precision, recall, fscore, average_precision, fscore_max,
                            fscore_max_taxo))
        elif state == 'semeval':
            logger.info(
                'Semeval Results: loss= {:.4f}, time: {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, Average_precision = {:.4f}, Average_max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                    .format(ave_loss, time.time() - t_t, precision, recall, fscore, average_precision, fscore_max,
                            fscore_max_taxo))
        elif state == 'semeval_trial':
            logger.info(
                'Semeval_trial Results: loss= {:.4f}, time: {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, Average_precision = {:.4f}, Average_max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                    .format(ave_loss, time.time() - t_t, precision, recall, fscore, average_precision, fscore_max,
                            fscore_max_taxo))
        else:
            logger.info('Error!')

    return fscore

################## TEST-Separately ##################
def test_separately(epoch, labels, state):
    model.eval()
    with torch.no_grad():

        for i in range(len(labels)):
            logger.info(len(labels[i][0]))
            key = labels[i][0]
            terms = torch.LongTensor(key).to(device=Config.device)
            e1 = torch.LongTensor(labels[i][1]).to(device=Config.device)
            e2 = torch.LongTensor(labels[i][2]).to(device=Config.device)
            rel = torch.LongTensor(labels[i][3]).to(device=Config.device)
            label = torch.FloatTensor(labels[i][4]).to(device=Config.device)
            taxo = labels[i][5]
            e1_index = torch.LongTensor(labels[i][6]).to(device=Config.device)
            e2_index = torch.LongTensor(labels[i][7]).to(device=Config.device)
            fre = torch.FloatTensor(labels[i][8]).to(device=Config.device)
            degree = torch.FloatTensor(labels[i][9]).to(device=Config.device)
            substr = torch.FloatTensor(labels[i][10]).to(device=Config.device)

            pred = model.forward(e1, e2, rel, X, adjs, terms, e1_index, e2_index, word_embs, fre, degree, substr)

            label = label.view(label.size()[0], 1)
            if loss_type == "F1":
                loss = f1_loss(pred, label)
            else:
                loss = model.loss(pred, label)

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

            if state == 'semeval_trial':
                logger.info(
                    'Semeval_T Taxo: loss= {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, BestF_Precision = {:.4f}, BestF_Recall= {:.4f}, Best_Threshold = {:.4f}, Average_precision = {:.4f}, Max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                        .format(loss.item(), p[1], r[1], f[1], Pre, Rec, threshold, average_p, f_max, f_max_taxo))
            elif state == 'semeval':
                logger.info(
                    'Semeval Taxo: loss= {:.4f}, Precision = {:.4f}, Recall= {:.4f}, Fscore = {:.4f}, BestF_Precision = {:.4f}, BestF_Recall= {:.4f}, Best_Threshold = {:.4f}, Average_precision = {:.4f}, Max_fscore = {:.4f}, Average_fscore_taxo = {:.4f}'
                        .format(loss.item(), p[1], r[1], f[1], Pre, Rec, threshold, average_p, f_max, f_max_taxo))
            else:
                logger.info("Error")

################## Visualization ##################
def visualization(labels, data_type):
    model.eval()
    with torch.no_grad():

        for i in range(len(labels)):
            logger.info(len(labels[i][0]))
            key = labels[i][0]
            terms = torch.LongTensor(key).to(device=Config.device)
            e1 = torch.LongTensor(labels[i][1]).to(device=Config.device)
            e2 = torch.LongTensor(labels[i][2]).to(device=Config.device)
            rel = torch.LongTensor(labels[i][3]).to(device=Config.device)
            label = torch.FloatTensor(labels[i][4]).to(device=Config.device)
            taxo = labels[i][5]
            e1_index = torch.LongTensor(labels[i][6]).to(device=Config.device)
            e2_index = torch.LongTensor(labels[i][7]).to(device=Config.device)
            fre = torch.FloatTensor(labels[i][8]).to(device=Config.device)
            degree = torch.FloatTensor(labels[i][9]).to(device=Config.device)
            substr = torch.FloatTensor(labels[i][10]).to(device=Config.device)

            pred = model.forward(e1, e2, rel, X, adjs, terms, e1_index, e2_index, word_embs, fre, degree, substr)
            pred = pred.cpu().detach().numpy().flatten()

            value = 0.5
            Y_matrix = np.where(pred > value, 1, 0)
            head = labels[i][1]
            tail = labels[i][2]

            if data_type == 'semeval':
                dst = "outputs/semeval_output_pairs_" + str(i) + ".txt"
            else:
                dst = "outputs/val_output_pairs_" + str(i) + ".txt"
            import re
            with open(dst, 'w') as file:
                for num in range(len(head)):
                    if Y_matrix[num] != 0:
                        row = head[num]
                        col = tail[num]
                        file.write(str(re.sub('_', ' ', id_word_map[row])) + '\t' + str(
                            re.sub('_', ' ', id_word_map[col])) + '\t' + str(pred[num]) + '\n')

############## Main ##################
def main():

    for epoch in range(epochs):
        train(epoch, train_labels)
        if (epoch + 1) % 1 == 0:
            F_score = test(epoch, val_labels, 'val')
        if F_score > 0.40 or epoch == (epochs-1):
            test(epoch, semeval_labels_RL, 'semeval')
            test_separately(epoch, semeval_labels_RL, 'semeval')
            # visualization(semeval_labels_RL, 'semeval')
            break

        #if (epoch + 1) % 100 == 0:
            # test(epoch, semeval_trial_labels, 'semeval_trial')
            # test(epoch, semeval_labels_RL, 'semeval')
            # test_separately(epoch, semeval_labels_RL, 'semeval')
            # test(epoch, semeval_labels, 'semeval')
            # test_separately(epoch, semeval_labels, 'semeval')

if __name__ == '__main__':
    main()




