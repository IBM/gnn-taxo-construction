import os
import re
import time
import shutil
import logging
import pandas as pd
import logging
import pandas as pd
from TaxoRL.code.utils_tree import read_tree_file, read_edge_files, load_candidate_from_pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
import datetime
import pickle
import codecs
import csv
import os
import re

logging.basicConfig(level=logging.INFO)

from morph import lemmatize
def get_freqs(self, hypo, hyper):
    hypo = str(hypo)
    hyper = str(hyper)
    hypo_variants = set([hypo, hypo.lower(), lemmatize(hypo), lemmatize(hypo).lower()])
    hyper_variants = set([hyper, hyper.lower(), lemmatize(hyper), lemmatize(hyper).lower()])
    freqs = [0]
    for w in hypo_variants:
        for iw in hyper_variants:
            if w in self._hypo2hyper and iw in self._hypo2hyper[w]: freqs.append(self._hypo2hyper[w][iw])
    return max(freqs)


def read_hh_pairs(file, terms_dict):
    with open('input_pairs_fre.txt', 'w') as output_file:
        logging.info('Reading file: {0}'.format(file))
        df = pd.read_csv(file, sep='\t', error_bad_lines = False, quoting=csv.QUOTE_NONE, encoding='utf-8')
        df.columns = ['hyponym', 'hypernym', 'relation', 'freq']
        count = 0
        for i, row in df.iterrows():

            hypo = str(row["hyponym"]).split("#")[0].lower()
            hyper = str(row["hypernym"]).split("#")[0].lower()
            hypo = re.sub(' ', '_', hypo)
            hyper = re.sub(' ', '_', hyper)
            freq = int(row["freq"])

            if hypo in terms_dict and hyper in terms_dict:
                output_file.write(str(hypo) + '\t' + str(hyper) + '\t' + 'is-a' + '\t' + str(freq) + '\n')
                count += 1
        print(count)

# Load the labels
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
                    hypo2hyper_edgeonly.append([hypo_, hyper_])
            trees.append([terms, hypo2hyper_edgeonly])
    return trees


def main():

    trees = read_tree_file(
        "./TaxoRL/datasets/wn-bo/wn-bo-trees-4-11-50-train533-lower.ptb",
        given_root=False, filter_root=False, allow_up=True)
    trees_val = read_tree_file(
        "./TaxoRL/datasets/wn-bo/wn-bo-trees-4-11-50-dev114-lower.ptb",
        given_root=False, filter_root=False, allow_up=True)
    trees_test = read_tree_file(
        "./TaxoRL/datasets/wn-bo/wn-bo-trees-4-11-50-test114-lower.ptb",
        given_root=False, filter_root=False, allow_up=True)
    trees_semeval = read_files('./TaxoRL/datasets/SemEval-2016/EN/',
                               given_root=True, filter_root=False, allow_up=False)
    trees_semeval_trial = read_files("./TaxoRL/datasets/SemEval-2016/trial/",
                                     given_root=True, filter_root=False, allow_up=False)

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

    train_tree = tree_no_intersect[:num]
    val_tree = tree_no_intersect[num:]

    with open('Trees.pkl', 'wb') as f:
        pickle.dump([train_tree, val_tree, trees_semeval, trees_semeval_trial, vocab_new], f)
    vocab = vocab_new

    terms_dict = {}
    count = 0
    for word in vocab:
        if word not in terms_dict:
            terms_dict[word] = count
            count = count + 1

    print('Num of all terms:', len(terms_dict))

    file = './taxi/taxi_pairs_all.txt'
    read_hh_pairs(file, terms_dict)

if __name__ == '__main__':
    main()
