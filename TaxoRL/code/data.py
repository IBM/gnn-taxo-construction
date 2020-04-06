import argparse
import os
#import _dynet as dy
import pickle
from collections import defaultdict
from tqdm import tqdm

#from model_RL import Policy
#from utils_tree import read_tree_file, read_edge_files, load_candidate_from_pickle
#from lstm_common import *
#from evaluation_common import *
#from knowledge_resource import KnowledgeResource
#from features import *
import time
import codecs

ap = argparse.ArgumentParser()
# file path parameters
ap.add_argument('--corpus_prefix', default='../corpus/3in1_twodatasets/3in1_twodatasets',
                help='path to the corpus resource')
ap.add_argument('--dataset_prefix', default='../datasets/wn-bo', help='path to the train/test/val/rel data')
ap.add_argument('--model_prefix_file', default='twodatasets_subseqFeat', help='where to store the result')
ap.add_argument('--embeddings_file', default='../../Wikipedia_Word2vec/glove.6B.50d.txt',
                help='path to word embeddings file')
ap.add_argument('--trainname', default='train_wnbo_hyper', help='name of training data')
ap.add_argument('--valname', default='dev_wnbo_hyper', help='name of val data')
ap.add_argument('--testname', default='test_wnbo_hyper', help='name of test data')
# dimension parameters
ap.add_argument('--NUM_LAYERS', default=2, help='number of layers of LSTM')
ap.add_argument('--HIST_LSTM_HIDDEN_DIM', default=60)
ap.add_argument('--POS_DIM', default=4)
ap.add_argument('--DEP_DIM', default=5)
ap.add_argument('--DIR_DIM', default=1)
ap.add_argument('--MLP_HIDDEN_DIM', default=60)
ap.add_argument('--PATH_LSTM_HIDDEN_DIM', default=60)
# model settings
ap.add_argument('--max_paths_per_pair', type=int, default=200,
                help='limit the number of paths per pair. Invalid when loading from pkl')
ap.add_argument('--gamma', default=0.4)
ap.add_argument('--n_rollout', type=int, default=10, help='run for each sample')
ap.add_argument('--lr', default=1e-3, help='learning rate')
ap.add_argument('--choose_max', default=True, help='choose action with max prob when testing')
ap.add_argument('--allow_up', default=True, help='allow to attach some term as new root')
ap.add_argument('--reward', default='edge', choices=['hyper', 'edge', 'binary', 'fragment'])
ap.add_argument('--reward_form', default='diff', choices=['last', 'per', 'diff'])
# ablation parameters
ap.add_argument('--allow_partial', default=True, help='allow only partial tree is built')
ap.add_argument('--use_freq_features', default=True, help='use freq features')
ap.add_argument('--use_features', default=True, help='use surface features')
ap.add_argument('--use_path', default=True, help='use path-based info')
ap.add_argument('--use_xy_embeddings', default=True, help='use word embeddings')
# misc
ap.add_argument('--test_semeval', default=True, help='run tests on semeval datasets')
ap.add_argument('--load_model_file', default=None,
                help='if not None, load model from a file')
ap.add_argument('--load_opt', default=False, help='load opt along with the loaded model')
# parameters that are OUTDATED. may or may not affect performance
ap.add_argument('--word_dropout_rate', default=0.25, help='replace a token with <unk> with specified probability')
ap.add_argument('--path_dropout_rate', default=0, help='dropout of LSTM path embedding')
ap.add_argument('--no_training', default=False, help='load sample trees for training')
ap.add_argument('--debug', default=False, help='debug or normal run')
ap.add_argument('--n_rollout_test', type=int, default=5, help='beam search width')
ap.add_argument('--discard_rate', default=0., help='discard a pair w.o path info by discard_rate')
ap.add_argument('--set_max_height', default=False, help='limit the max height of tree')
ap.add_argument('--use_height_ebd', default=False, help='consider the height of each node')
ap.add_argument('--use_history', default=False, help='use history of taxonomy construction')
ap.add_argument('--use_sibling', default=False, help='use sibling signals')
ap.add_argument('--require_info', default=False, help='require there has to be info to infer...')
ap.add_argument('--given_root_train', default=False, help='[outdated]give gold root or not')
ap.add_argument('--given_root_test', default=False, help='[outdated]give gold root or not')
ap.add_argument('--filter_root', default=False, help='[outdated]filter root by term counts')
ap.add_argument('--one_layer', default=False, help='only one layer after pair representation')
ap.add_argument('--update_word_ebd', default=False, help='update word embedding or use fixed pre-train embedding')
ap.add_argument('--use_candidate', default=True, help='use candidates instead of considering all remaining pairs')
ap.add_argument('--height_ebd_dim', default=30)
args = ap.parse_args()

#from utils_common import check_error, update_best, get_micro_f1, check_data, load_paths_and_word_vectors, \
#    get_vocabulary, print_config, save_path_info, test, check_error_np

opt = vars(args)
score_filename = 'pickled_data/path{}_roll{}_debug{}.pkl'.format(args.max_paths_per_pair, args.n_rollout, args.debug)
n_run = 1
while os.path.exists(score_filename):
    score_filename = score_filename[:-len(str(n_run - 1))] + str(n_run)
    n_run += 1
print('score_filename', score_filename)
print('start time:{}'.format(time.ctime()))
print('last modified:{}'.format(time.ctime(os.path.getmtime(__file__))))

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


def main():
    #print_config(opt)
    # Load the relations
    with codecs.open(args.dataset_prefix + '/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = {relation: i for i, relation in enumerate(relations)}

    # Load the datasets

    trainname = '/' + args.trainname + '.tsv'
    valname = '/' + args.valname + '.tsv'
    testname = '/' + args.testname + '.tsv'
    print('Loading the dataset...', trainname, '*' * 10)
    train_set = load_dataset(args.dataset_prefix + trainname, relations)
    print('Loading the dataset...', valname, '*' * 10)
    val_set = load_dataset(args.dataset_prefix + valname, relations)
    print('Loading the dataset...', testname, '*' * 10)
    test_set = load_dataset(args.dataset_prefix + testname, relations)
    # y_train = [relation_index[label] for label in train_set.values()]
    # y_val = [relation_index[label] for label in val_set.values()]
    # y_test = [relation_index[label] for label in test_set.values()]
    dataset_keys = list(train_set.keys()) + list(val_set.keys()) + list(test_set.keys())
    # add (x, root) to dataset_keys
    vocab = set()
    for (x, y) in dataset_keys:
        vocab.add(x)
        vocab.add(y)
    dataset_keys += [(term, 'root007') for term in vocab]

    # Load the resource (processed corpus)
    #print('Loading the corpus...', args.corpus_prefix, '*' * 10)
    #corpus = KnowledgeResource(args.corpus_prefix)



    #(word_vectors, word_index, word_set, dataset_instances, pos_index, dep_index, dir_index, pos_inverted_index,
    # dep_inverted_index, dir_inverted_index) = pickle.load(open('../../preload_data_3in1_subseqFeat_debugFalse.pkl', 'rb'))
    import pandas as pd
    dataset_instances = pd.read_pickle('../../dataset_instances_twodatasets.pkl')
    word_index = pd.read_pickle('word_index_twodatasets.pkl')

    #print('Number of words %d, number of pos tags: %d, number of dependency labels: %d, number of directions: %d' % \
    #      (len(word_index), len(pos_index), len(dep_index), len(dir_index)))

    # dataset_instances is now (paths, x_y_vectors, features)
    X_train = dataset_instances[:len(train_set)]
    X_val = dataset_instances[len(train_set):len(train_set) + len(val_set)]
    X_test = dataset_instances[len(train_set) + len(val_set):]
    print(len(X_train), len(X_val), len(X_test))



if __name__ == '__main__':
    main()
