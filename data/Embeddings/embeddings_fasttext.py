
''''
Embeddings from FastText
'''

from __future__ import division
from __future__ import print_function
import pickle
import pandas as pd
from pyfasttext import FastText

file_name = 'adj_10.pkl'
(data, rows, columns, vocab_dict) = pd.read_pickle(file_name)
d = {}

# cc.en.300.bin can be got from: https://fasttext.cc/docs/en/crawl-vectors.html
model = FastText('cc.en.300.bin')
for word in vocab_dict:
    if word not in d:
        d[word] = model[word]

with open('embeddings_fasttaxt.pkl', 'wb') as f:
    pickle.dump(d, f)