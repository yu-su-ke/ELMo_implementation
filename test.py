import time

import pandas as pd
from pathlib import Path
from janome.tokenizer import Tokenizer

from function import prepare_labels
from format_text import preprocess


DATA_ROOT = './'
train = pd.read_csv(Path(DATA_ROOT + 'livedoor') / "train.csv", header=None)
train_num = 500

start = time.time()
train_path = [train[0][i] for i in range(train_num)]
train_document = [open(DATA_ROOT + 'livedoor/' + train_path[j], 'r', encoding='utf-8').read() for j in range(train_num)]
j_t = Tokenizer()
x_train = []
for train_text in train_document:
    x_train.append(preprocess(train_text, j_t))
x_train = pd.Series(x_train)
print(x_train)
print(type(x_train))
elapsed_time = time.time() - start
print('elapsed_time:{0}'.format(elapsed_time) + '[sec]')

"""
start = time.time()

train_path = [train[0][i] for i in range(train_num)]
train_document = (open(DATA_ROOT + 'livedoor/' + train_path[j], 'r', encoding='utf-8').read() for j in range(train_num))
j_t = Tokenizer()
x_train = []
for train_text in train_document:
    x_train.append(preprocess(train_text, j_t))
x_train = pd.Series(x_train)
print(x_train)

elapsed_time = time.time() - start
