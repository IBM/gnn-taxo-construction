''''
Data Preprocessing
'''

from __future__ import division
from __future__ import print_function
from data.TaxoRL_dataset.utils_tree import read_tree_file
from collections import OrderedDict
import pickle
import codecs
import os
import re
from scipy.sparse import coo_matrix
import numpy as np
import os.path
from difflib import SequenceMatcher
from utils import Endswith, Contains, Prefix_match, Suffix_match, LCS, LD

# Read the Semeval Taxonomies
def read_files(in_path, given_root=False, filter_root=False, allow_up=True, noUnderscore=False):
    trees = []
    for root, dirs, files in os.walk(in_path):
        for filename in files:
            if not filename.endswith('taxo'):
                continue
            file_path = root + filename
            print('read_edge_files', file_path)
            with codecs.open(file_path, 'r', 'utf-8') as f:
                hypo2hyper_edgeonly = []
                terms = set()
                for line in f:
                    hypo, hyper = line.strip().lower().split('\t')[1:]
                    hypo_ = re.sub(' ', '_', hypo)
                    hyper_ = re.sub(' ', '_', hyper)
                    terms.add(hypo_)
                    terms.add(hyper_)
                    hypo2hyper_edgeonly.append([hypo_,hyper_])
            print(len(terms))

            trees.append([terms, hypo2hyper_edgeonly, filename])
    return trees

# Load the data from dataset_file
def load_dataset(dataset_file, relations):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = [tuple(line.strip().split('\t')) for line in f_in]
        dataset = {(x.lower(), y.lower()): relation for (x, y, relation) in dataset if relation in relations}
    return dataset


def LCS_indict(x, y):
    match = SequenceMatcher(None, x, y).find_longest_match(0, len(x), 0, len(y))
    if x[match.a:match.a+match.size] in terms_dict:
        return 1
    else:
        return 0

################### Get the labels/taxonomies ######################
# Read all taxonomies
trees = read_tree_file(
    "./data/TaxoRL_dataset/wn-bo/wn-bo-trees-4-11-50-train533-lower.ptb",
    given_root=False, filter_root=False, allow_up=True)
trees_val = read_tree_file(
    "./data/TaxoRL_dataset/wn-bo/wn-bo-trees-4-11-50-dev114-lower.ptb",
    given_root=False, filter_root=False, allow_up=True)
trees_test = read_tree_file(
    "./data/TaxoRL_dataset/wn-bo/wn-bo-trees-4-11-50-test114-lower.ptb",
    given_root=False, filter_root=False, allow_up=True)
trees_semeval = read_files('./data/TaxoRL_dataset/SemEval-2016/EN/',
                           given_root=True, filter_root=False, allow_up=False)
trees_semeval_trial = read_files("./data/TaxoRL_dataset/SemEval-2016/trial/",
                                 given_root=True, filter_root=False, allow_up=False)

# Build the vocabulary
vocab = set()
for i in range(len(trees)):
    vocab = vocab.union(trees[i].terms)
for i in range(len(trees_val)):
    vocab = vocab.union(trees_val[i].terms)
for i in range(len(trees_test)):
    vocab = vocab.union(trees_test[i].terms)
print('size of terms in training:', len(vocab))

for i in range(len(trees_semeval)):
    vocab = vocab.union(trees_semeval[i][0])
print('size of terms in the semeval:', len(vocab))
for i in range(len(trees_semeval_trial)):
    vocab = vocab.union(trees_semeval_trial[i][0])
print('size of terms added trial:', len(vocab))

vocab_semeval = set()
for i in range(len(trees_semeval)):
    vocab_semeval = vocab_semeval.union(trees_semeval[i][0])
print('size of terms (semeval):', len(vocab_semeval))

# Remove the overlapping taxonomies.
tree_no_intersect = []
count = 0
falsecount = 0
for i in range(len(trees)):
    if len(trees[i].terms & vocab_semeval) == 0:
        count = count + 1
        tree_no_intersect.append(trees[i])
    else:
        falsecount = falsecount + 1
for i in range(len(trees_val)):
    if len(trees_val[i].terms & vocab_semeval) == 0:
        count = count + 1
        tree_no_intersect.append(trees_val[i])
    else:
        falsecount = falsecount + 1

for i in range(len(trees_test)):
    if len(trees_test[i].terms & vocab_semeval) == 0:
        count = count + 1
        tree_no_intersect.append(trees_test[i])
    else:
        falsecount = falsecount + 1
print('num of trees which has no intersaction with label taxos:', count)
print('Trees need to be removed:', falsecount)

# Get the new vocabulary.
num = int(len(tree_no_intersect) * 0.8)
vocab_new = set()
for i in range(len(tree_no_intersect)):
    vocab_new = vocab_new.union(tree_no_intersect[i].terms)
print('size of terms in filted trees:', len(vocab_new))
for i in range(len(trees_semeval)):
    vocab_new = vocab_new.union(trees_semeval[i][0])
print('size of terms in filted trees + semeval:', len(vocab_new))
for i in range(len(trees_semeval_trial)):
    vocab_new = vocab_new.union(trees_semeval_trial[i][0])
print('size of terms added trial:', len(vocab_new))
train_trees = tree_no_intersect[:num]
val_trees = tree_no_intersect[num:]
vocab = vocab_new

# Build the dictionary
terms_dict = {}
count = 0
for word in vocab:
    if word not in terms_dict:
        terms_dict[word] = count
        count = count + 1

print('Num of all terms:',len(terms_dict))
index_word_map = {v: k for k, v in terms_dict.items()}


############### Get the adjacency matrix #################
# If option == 1, use the TAXI extracted pairs as input.
# If option == 2, use the pairs in TaxoRL as input.
option = 1
data = []
rows = []
columns = []
if option == 1:
    with open('./data/TAXI_dataset/input_pairs_fre.txt', 'r') as ft:
        for x in ft.readlines():
            head, tail, relation, score = x.strip().split('\t')
            if head in terms_dict and tail in terms_dict and head != tail:
                data.append(float(score))
                rows.append(terms_dict[head])
                columns.append(terms_dict[tail])
    print('Num of edges:')
    print(len(data))

elif option == 2:
    adj_dic = {}
    with open('./data/TAXI_dataset/input_pairs_fre.txt', 'r') as ft:
        for x in ft.readlines():
            head, tail, relation, score = x.strip().split('\t')
            if head in terms_dict and tail in terms_dict:
                pair = (terms_dict[head], terms_dict[tail])
                if pair not in adj_dic:
                    adj_dic[pair] = float(score)
                else:
                    adj_dic[pair] = adj_dic[pair] + float(score)

    with codecs.open('./data/TaxoRL_dataset/wn-bo/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = {relation: i for i, relation in enumerate(relations)}

    # Load the datasets
    trainname = './data/TaxoRL_dataset/wn-bo/train_wnbo_hyper.tsv'
    valname = './data/TaxoRL_dataset/wn-bo/dev_wnbo_hyper.tsv'
    testname = './data/TaxoRL_dataset/wn-bo/test_wnbo_hyper.tsv'
    print('Loading the dataset...', trainname, '*' * 10)
    train_set = load_dataset(trainname, relations)
    print('Loading the dataset...', valname, '*' * 10)
    val_set = load_dataset(valname, relations)
    print('Loading the dataset...', testname, '*' * 10)
    test_set = load_dataset(testname, relations)

    dataset_keys = list(train_set.keys()) + list(val_set.keys()) + list(test_set.keys())

    for item in dataset_keys:
        head = item[0]
        tail = item[1]
        if head in terms_dict and tail in terms_dict and head != tail:
            pair = (terms_dict[head], terms_dict[tail])
            if pair not in adj_dic:
                data.append(1)
            else:
                data.append(adj_dic[pair])
            rows.append(terms_dict[head])
            columns.append(terms_dict[tail])

# Substring
for row in vocab:
    for col in vocab:
        if col in row and row != col and len(col) > 3:
            data.append(100) # set frequency to 100 if col in row.
            rows.append(terms_dict[row])
            columns.append(terms_dict[col])

# Save adjacency matrix
with open('adj_input.pkl', 'wb') as f:
    pickle.dump([data, rows, columns, terms_dict], f)
print("Saved adj.pkl")

# Sparse Matrix
rel_list = ['ISA']
num_entities = len(terms_dict)
num_relations = len(rel_list)
adj = coo_matrix((data, (rows, columns)), shape=(num_entities, num_entities)).toarray()
adj = np.where(adj >= 10, adj, 0) # 10: threshold of frequency.
print("Finished the data preparation")

# preprocess the input data into batches
def build_batch(trees, batch = 20):
    train_labels = []
    tree_dict = {}
    print(len(trees))
    for i_episode in range(len(trees)):

        if i_episode != (len(trees)-1):
            if i_episode % batch != 0 or i_episode == 0:
                T = trees[i_episode]
                for k, v in T.taxo.items():
                    if v == 'root007':
                        continue
                    head = terms_dict[k]
                    tail = terms_dict[v]
                    if head in tree_dict:
                        tree_dict[head].append(tail)
                    else:
                        tree_dict[head] = [tail]
                    if tail not in tree_dict:
                        tree_dict[tail] = []
            else:
                if i_episode != (len(trees)-1):
                    tree_dict = OrderedDict(sorted(tree_dict.items(), key=lambda t: t[0]))
                    keys = np.array(list(tree_dict.keys()))
                    num_keys = len(keys)
                    input_mat = [[0 for x in range(num_keys)] for y in range(num_keys)]

                    for k, v in tree_dict.items():
                        x = np.where(keys == k)[0][0]
                        for i in range(len(v)):
                            y = np.where(keys == v[i])[0][0]
                            input_mat[x][y] = 1

                    input_mat = np.array(input_mat)
                    rowsum_label = input_mat.sum(axis=1)
                    rootnode = np.where(rowsum_label == 0)[0]

                    row_idx = np.array(keys)
                    col_idx = np.array(keys)
                    mask = adj[row_idx, :][:, col_idx]

                    deg = []
                    colsum = mask.sum(axis=0)
                    rowsum = mask.sum(axis=1)
                    for j in range(0, len(mask)):
                        deg.append([colsum[j], rowsum[j]])

                    mask_one = np.where(mask > 0, 1, 0)
                    deg_one = []
                    colsum_one = mask_one.sum(axis=0)
                    rowsum_one = mask_one.sum(axis=1)
                    for j in range(0, len(mask_one)):
                        deg_one.append([colsum_one[j], rowsum_one[j]])

                    head_array = []
                    tail_array = []
                    label_array = []
                    head_index = []
                    tail_index = []
                    fre_array = []
                    degree = []
                    substr = []
                    for r in range(num_keys):
                        for c in range(num_keys):
                            if mask[r][c] != 0 and r != c:
                                head_array.append(keys[r])
                                head_index.append(r)
                                tail_array.append(keys[c])
                                tail_index.append(c)
                                label_array.append(input_mat[r][c])
                                fre_array.append([mask[r][c]])
                                degree.append(
                                    [deg[r][0], deg[r][1], deg[c][0], deg[c][1], deg_one[r][0], deg_one[r][1],
                                     deg_one[c][0],
                                     deg_one[c][1]])
                                s1 = Endswith(index_word_map[keys[c]], index_word_map[keys[r]])
                                s2 = Contains(index_word_map[keys[c]], index_word_map[keys[r]])
                                s3 = Prefix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                                s4 = Suffix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                                s5 = LCS(index_word_map[keys[c]], index_word_map[keys[r]])
                                s6 = LD(index_word_map[keys[c]], index_word_map[keys[r]])
                                s7 = LCS_indict(index_word_map[keys[c]], index_word_map[keys[r]])
                                substr.append([s1, s2, s3, s4, s5, s6, s7])

                    rel = [0 for i in range(len(head_array))]
                    train_labels.append([keys, head_array, tail_array, rel, label_array, input_mat, head_index, tail_index, fre_array, degree, substr, rootnode])
                    tree_dict = {}
                    T = trees[i_episode]
                    for k, v in T.taxo.items():
                        if v == 'root007':
                            continue
                        head = terms_dict[k]
                        tail = terms_dict[v]

                        if head in tree_dict:
                            tree_dict[head].append(tail)
                        else:
                            tree_dict[head] = [tail]
                        if tail not in tree_dict:
                            tree_dict[tail] = []
        else:
            if i_episode % batch == 0:
                tree_dict = OrderedDict(sorted(tree_dict.items(), key=lambda t: t[0]))
                keys = np.array(list(tree_dict.keys()))
                num_keys = len(keys)
                input_mat = [[0 for x in range(num_keys)] for y in range(num_keys)]

                for k, v in tree_dict.items():
                    x = np.where(keys == k)[0][0]
                    for i in range(len(v)):
                        y = np.where(keys == v[i])[0][0]
                        input_mat[x][y] = 1

                input_mat = np.array(input_mat)
                rowsum_label = input_mat.sum(axis=1)
                rootnode = np.where(rowsum_label == 0)[0]

                row_idx = np.array(keys)
                col_idx = np.array(keys)
                mask = adj[row_idx, :][:, col_idx]

                deg = []
                colsum = mask.sum(axis=0)
                rowsum = mask.sum(axis=1)
                for j in range(0, len(mask)):
                    deg.append([colsum[j], rowsum[j]])

                mask_one = np.where(mask > 0, 1, 0)
                deg_one = []
                colsum_one = mask_one.sum(axis=0)
                rowsum_one = mask_one.sum(axis=1)
                for j in range(0, len(mask_one)):
                    deg_one.append([colsum_one[j], rowsum_one[j]])

                head_array = []
                tail_array = []
                label_array = []
                head_index = []
                tail_index = []
                fre_array = []
                degree = []
                substr = []
                for r in range(num_keys):
                    for c in range(num_keys):
                        if mask[r][c] != 0 and r != c:
                            head_array.append(keys[r])
                            head_index.append(r)
                            tail_array.append(keys[c])
                            tail_index.append(c)
                            label_array.append(input_mat[r][c])
                            fre_array.append([mask[r][c]])

                            degree.append(
                                [deg[r][0], deg[r][1], deg[c][0], deg[c][1], deg_one[r][0], deg_one[r][1],
                                 deg_one[c][0],
                                 deg_one[c][1]])

                            s1 = Endswith(index_word_map[keys[c]], index_word_map[keys[r]])
                            s2 = Contains(index_word_map[keys[c]], index_word_map[keys[r]])
                            s3 = Prefix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                            s4 = Suffix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                            s5 = LCS(index_word_map[keys[c]], index_word_map[keys[r]])
                            s6 = LD(index_word_map[keys[c]], index_word_map[keys[r]])
                            s7 = LCS_indict(index_word_map[keys[c]], index_word_map[keys[r]])
                            substr.append([s1, s2, s3, s4, s5, s6, s7])

                rel = [0 for i in range(len(head_array))]
                train_labels.append([keys, head_array, tail_array, rel, label_array, input_mat, head_index, tail_index, fre_array, degree, substr, rootnode])

                tree_dict = {}
                T = trees[i_episode]
                for k, v in T.taxo.items():
                    if v == 'root007':
                        continue
                    head = terms_dict[k]
                    tail = terms_dict[v]

                    if head in tree_dict:
                        tree_dict[head].append(tail)
                    else:
                        tree_dict[head] = [tail]
                    if tail not in tree_dict:
                        tree_dict[tail] = []

            tree_dict = OrderedDict(sorted(tree_dict.items(), key=lambda t: t[0]))
            keys = np.array(list(tree_dict.keys()))
            num_keys = len(keys)
            input_mat = [[0 for x in range(num_keys)] for y in range(num_keys)]

            for k, v in tree_dict.items():
                x = np.where(keys == k)[0][0]
                for i in range(len(v)):
                    y = np.where(keys == v[i])[0][0]
                    input_mat[x][y] = 1

            input_mat = np.array(input_mat)
            rowsum_label = input_mat.sum(axis=1)
            rootnode = np.where(rowsum_label == 0)[0]

            row_idx = np.array(keys)
            col_idx = np.array(keys)
            mask = adj[row_idx, :][:, col_idx]

            deg = []
            colsum = mask.sum(axis=0)
            rowsum = mask.sum(axis=1)
            for j in range(0, len(mask)):
                deg.append([colsum[j], rowsum[j]])

            mask_one = np.where(mask > 0, 1, 0)
            deg_one = []
            colsum_one = mask_one.sum(axis=0)
            rowsum_one = mask_one.sum(axis=1)
            for j in range(0, len(mask_one)):
                deg_one.append([colsum_one[j], rowsum_one[j]])

            head_array = []
            tail_array = []
            label_array = []
            head_index = []
            tail_index = []
            fre_array = []
            degree = []
            substr = []
            for r in range(num_keys):
                for c in range(num_keys):
                    if mask[r][c] != 0 and r != c:
                        head_array.append(keys[r])
                        head_index.append(r)
                        tail_array.append(keys[c])
                        tail_index.append(c)
                        label_array.append(input_mat[r][c])
                        fre_array.append([mask[r][c]])
                        degree.append(
                            [deg[r][0], deg[r][1], deg[c][0], deg[c][1], deg_one[r][0], deg_one[r][1], deg_one[c][0],
                             deg_one[c][1]])
                        s1 = Endswith(index_word_map[keys[c]], index_word_map[keys[r]])
                        s2 = Contains(index_word_map[keys[c]], index_word_map[keys[r]])
                        s3 = Prefix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                        s4 = Suffix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                        s5 = LCS(index_word_map[keys[c]], index_word_map[keys[r]])
                        s6 = LD(index_word_map[keys[c]], index_word_map[keys[r]])
                        s7 = LCS_indict(index_word_map[keys[c]], index_word_map[keys[r]])
                        substr.append([s1, s2, s3, s4, s5, s6, s7])

            rel = [0 for i in range(len(head_array))]
            train_labels.append([keys, head_array, tail_array, rel, label_array, input_mat, head_index, tail_index, fre_array, degree, substr, rootnode])
            print('finished')
    return train_labels

# preprocess the input from TaxoRL and prepare the features
def preprocess_RL(trees, type):
    labels = []

    print(len(trees))
    for i_episode in range(len(trees)):
        T = trees[i_episode]
        try:
            hyper2hypo_w_freq = pickle.load(
                open('./data/TaxoRL_dataset/SemEval-2016/candidates_taxi/{}.pkl'.format(T[2] + '.candidate_w_freq'),
                     'rb'))
        except:
            print("Not privide taxo", T[2])
            continue

        tree_dict = {}
        for i in range(len(T[1])):

            if T[1][i][1] == 'root007':
                continue
            head = terms_dict[T[1][i][0]]
            tail = terms_dict[T[1][i][1]]

            if head in tree_dict:
                tree_dict[head].append(tail)
            else:
                tree_dict[head] = [tail]
            if tail not in tree_dict:
                tree_dict[tail] = []

        tree_dict = OrderedDict(sorted(tree_dict.items(), key=lambda t: t[0]))
        keys = np.array(list(tree_dict.keys()))
        num_keys = len(keys)
        input_mat = [[0 for x in range(num_keys)] for y in range(num_keys)]

        for k, v in tree_dict.items():
            x = np.where(keys == k)[0][0]
            for i in range(len(v)):
                y = np.where(keys == v[i])[0][0]
                input_mat[x][y] = 1
        input_mat = np.array(input_mat)
        rowsum_label = input_mat.sum(axis=1)
        rootnode = np.where(rowsum_label == 0)[0]

        mask = [[0 for x in range(num_keys)] for y in range(num_keys)]
        for i in range(len(T[1])):
            if T[1][i][1] == 'root007':
                continue
            if T[1][i][1] in T[1][i][0]:
                x = np.where(keys == terms_dict[T[1][i][0]])[0][0]
                y = np.where(keys == terms_dict[T[1][i][1]])[0][0]
                mask[x][y] = 1
        num_pairs = 0
        num_truepairs = 0
        num_20 = 0
        num_20_truepairs = 0
        for hyper in hyper2hypo_w_freq:
            for hypo in hyper2hypo_w_freq[hyper]:
                num_pairs += 1
                if hyper2hypo_w_freq[hyper][hypo] >= 20:
                    num_20 += 1
                    x = np.where(keys == terms_dict[hypo])[0][0]
                    y = np.where(keys == terms_dict[hyper])[0][0]
                    mask[x][y] = hyper2hypo_w_freq[hyper][hypo]
                    if input_mat[np.where(keys == terms_dict[hypo])[0][0]][np.where(keys == terms_dict[hyper])[0][0]] == 1:
                        num_20_truepairs += 1
                if input_mat[np.where(keys == terms_dict[hypo])[0][0]][np.where(keys == terms_dict[hyper])[0][0]] == 1:
                    num_truepairs += 1
        print(num_pairs, num_20, num_truepairs, num_20_truepairs)
        mask = np.array(mask)
        deg = []
        colsum = mask.sum(axis=0)
        rowsum = mask.sum(axis=1)
        for j in range(0, len(mask)):
            deg.append([colsum[j], rowsum[j]])
        hypo_zero = np.where(rowsum == 0)[0]
        hypo_arr = []
        for num_hypo in hypo_zero:
            if colsum[num_hypo] == 0:
                hypo_arr.append(num_hypo)

        mask_one = np.where(mask > 0, 1, 0)
        deg_one = []
        colsum_one = mask_one.sum(axis=0)
        rowsum_one = mask_one.sum(axis=1)
        for j in range(0, len(mask_one)):
            deg_one.append([colsum_one[j], rowsum_one[j]])

        head_array = []
        tail_array = []
        label_array = []
        head_index = []
        tail_index = []
        fre_array = []
        degree = []
        substr = []
        for r in range(num_keys):
            for c in range(num_keys):
                if mask[r][c] != 0 and r != c:
                    head_array.append(keys[r])
                    head_index.append(r)
                    tail_array.append(keys[c])
                    tail_index.append(c)
                    label_array.append(input_mat[r][c])
                    fre_array.append([mask[r][c]])
                    degree.append(
                        [deg[r][0], deg[r][1], deg[c][0], deg[c][1], deg_one[r][0], deg_one[r][1], deg_one[c][0],
                         deg_one[c][1]])
                    s1 = Endswith(index_word_map[keys[c]], index_word_map[keys[r]])
                    s2 = Contains(index_word_map[keys[c]], index_word_map[keys[r]])
                    s3 = Prefix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                    s4 = Suffix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                    s5 = LCS(index_word_map[keys[c]], index_word_map[keys[r]])
                    s6 = LD(index_word_map[keys[c]], index_word_map[keys[r]])
                    s7 = LCS_indict(index_word_map[keys[c]], index_word_map[keys[r]])
                    substr.append([s1, s2, s3, s4, s5, s6, s7])

        print("RL data:")
        print("num of nodes:", num_keys)
        m1 = np.array(input_mat)
        print(np.count_nonzero(m1))
        m2 = np.array(mask)
        print(np.count_nonzero(m2))
        m3 = np.multiply(m1, m2)
        print(np.count_nonzero(m3))

        rel = [0 for i in range(len(head_array))]
        labels.append([keys, head_array, tail_array, rel, label_array, input_mat, head_index, tail_index, fre_array, degree, substr, rootnode])

    return labels

# preprocess the input data from Semeval and prepare the features
def preprocess(trees, type):
    labels = []

    print(len(trees))
    for i_episode in range(len(trees)):

        T = trees[i_episode]
        tree_dict = {}
        for i in range(len(T[1])):
            if T[1][i][1] == 'root007':
                continue
            head = terms_dict[T[1][i][0]]
            tail = terms_dict[T[1][i][1]]
            if head in tree_dict:
                tree_dict[head].append(tail)
            else:
                tree_dict[head] = [tail]
            if tail not in tree_dict:
                tree_dict[tail] = []

        tree_dict = OrderedDict(sorted(tree_dict.items(), key=lambda t: t[0]))
        keys = np.array(list(tree_dict.keys()))
        num_keys = len(keys)
        input_mat = [[0 for x in range(num_keys)] for y in range(num_keys)]
        for k, v in tree_dict.items():
            x = np.where(keys == k)[0][0]
            for i in range(len(v)):
                y = np.where(keys == v[i])[0][0]
                input_mat[x][y] = 1

        input_mat = np.array(input_mat)
        rowsum_label = input_mat.sum(axis=1)
        rootnode = np.where(rowsum_label == 0)[0]
        row_idx = np.array(keys)
        col_idx = np.array(keys)
        mask = adj[row_idx, :][:, col_idx]

        deg = []
        colsum = mask.sum(axis=0)
        rowsum = mask.sum(axis=1)
        for j in range(0, len(mask)):
            deg.append([colsum[j], rowsum[j]])

        mask_one = np.where(mask > 0, 1, 0)
        deg_one = []
        colsum_one = mask_one.sum(axis=0)
        rowsum_one = mask_one.sum(axis=1)
        for j in range(0, len(mask_one)):
            deg_one.append([colsum_one[j], rowsum_one[j]])

        head_array = []
        tail_array = []
        label_array = []
        head_index = []
        tail_index = []
        fre_array = []
        degree = []
        substr = []
        for r in range(num_keys):
            for c in range(num_keys):
                if mask[r][c] != 0 and r != c:
                    head_array.append(keys[r])
                    head_index.append(r)
                    tail_array.append(keys[c])
                    tail_index.append(c)
                    label_array.append(input_mat[r][c])
                    fre_array.append([mask[r][c]])
                    degree.append(
                        [deg[r][0], deg[r][1], deg[c][0], deg[c][1], deg_one[r][0], deg_one[r][1], deg_one[c][0],
                         deg_one[c][1]])
                    s1 = Endswith(index_word_map[keys[c]], index_word_map[keys[r]])
                    s2 = Contains(index_word_map[keys[c]], index_word_map[keys[r]])
                    s3 = Prefix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                    s4 = Suffix_match(index_word_map[keys[c]], index_word_map[keys[r]])
                    s5 = LCS(index_word_map[keys[c]], index_word_map[keys[r]])
                    s6 = LD(index_word_map[keys[c]], index_word_map[keys[r]])
                    s7 = LCS_indict(index_word_map[keys[c]], index_word_map[keys[r]])
                    substr.append([s1, s2, s3, s4, s5, s6, s7])

        print("data:")
        print("num of nodes:", num_keys)
        m1 = np.array(input_mat)
        print(np.count_nonzero(m1))
        m2 = np.array(mask)
        print(np.count_nonzero(m2))
        m3 = np.multiply(m1, m2)
        print(np.count_nonzero(m3))
        print("OOV:")
        mask = np.array(mask)
        input_mat = np.array(input_mat)
        print(np.sum(~mask.any(1)), np.sum(~input_mat.any(1)))

        rel = [0 for i in range(len(head_array))]
        labels.append([keys, head_array, tail_array, rel, label_array, input_mat, head_index, tail_index, fre_array, degree, substr, rootnode])

    return labels


############## Main ##################
def main():
    train_labels = build_batch(train_trees, 10)
    val_labels = build_batch(val_trees, 10)
    semeval_labels_RL = preprocess_RL(trees_semeval, 'semeval')
    semeval_labels = preprocess(trees_semeval, 'semeval')
    semeval_trial_labels = preprocess(trees_semeval_trial, 'semeval')

    with open('labels_input.pkl', 'wb') as f:
        pickle.dump([train_labels, val_labels, semeval_labels, semeval_labels_RL, semeval_trial_labels], f)

if __name__ == '__main__':
    main()
