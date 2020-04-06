import torch
from global_config import Config
import pandas as pd
from difflib import SequenceMatcher
import os
import re
import codecs

# Loss Function
class F1_Loss(torch.nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon
        self.beta = Config.F_beta

    def forward(self, probas, target):
        TP = torch.sum(probas.mul(target))
        precision = TP / (torch.sum(probas) + self.epsilon)
        recall = TP / (torch.sum(target) + self.epsilon)
        f1 = (self.beta**2 + 1) * precision * recall / ((self.beta**2) * precision + recall + self.epsilon)
        return 1 - f1.mean()

# Load embeddings
def load_embeddings(vocab_dict):
    embedding_file = './Embeddings/embeddings_fasttaxt.pkl'
    word_embeddings = pd.read_pickle(embedding_file)
    selected_word_vectors = []
    for k, v in vocab_dict.items():
        if k in word_embeddings:
            selected_word_vectors.append(word_embeddings[k])
        else:
            # print('No embedding for the word!')
            pass
    word_embs = torch.FloatTensor(selected_word_vectors).to(device=Config.device)
    return word_embs

# Logger initialization
def logger_init():
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    fhlr = logging.FileHandler(Config.output_name, mode='w')
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    logger.info(
        "output_name:%s, dropout_rate:%f, input_dropout:%f, channels:%d, init_emb_size:%d, embedding_dim:%d, learning_rate:%f, random_seed: %d, gc1_emb_size:%d, gc2_emb_size:%d, F_beta:%d",
        Config.output_name, Config.dropout_rate, Config.input_dropout, Config.channels, Config.init_emb_size,
        Config.embedding_dim,
        Config.learning_rate, Config.random_seed, Config.gc1_emb_size, Config.gc2_emb_size, Config.F_beta)
    #model_name = '{2}_{0}_{1}'.format(Config.input_dropout, Config.dropout, Config.model_name)

    return logger

def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

def Endswith(x, y):
    return int(y.endswith(x))

def Startswith(x, y):
    return int(y.startswith(x))

def Contains(x, y):
    return int(x in y)

def Prefix_match(x, y):
    k = 7
    for i in range(k):
        if x[:i+1] != y[:i+1]:
            return i
    return k

def Suffix_match(x, y):
    k = 7
    for i in range(k):
        if x[-i - 1:] != y[-i - 1:]:
            return i
    return k

def LCS(x, y):
    match = SequenceMatcher(None, x, y).find_longest_match(0, len(x), 0, len(y))
    res = 2.0 * match.size / (len(x) + len(y))  # [0, 1]
    return int(round(res, 1) * 10)  # [0,10]


def LD(x, y):
    res = 2.0 * (len(x) - len(y)) / (len(x) + len(y))  # (-2,2)
    return int(round(res, 1) * 10 + 20)  # [0, 40]

