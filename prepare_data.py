import pandas as pd

import numpy as np
from pathlib import Path
from janome.tokenizer import Tokenizer
from keras.preprocessing import text, sequence

from function import prepare_labels
from format_text import preprocess


def prepare_data(file_path, num=0):
    train = pd.read_csv(file_path, header=None)
    # data_num = train.shape[0]
    data_num = 100
    train_path = [train[0][i] for i in range(data_num)]
    train_document = [open('./livedoor/' + train_path[j], 'r', encoding='utf-8').read() for j in
                      range(data_num)]

    j_t = Tokenizer()

    x_document = []
    for train_text in train_document:
        x_document.append(preprocess(train_text, j_t))
    x_document = pd.Series(x_document)

    x_document.index = [i for i in range(num, num + data_num)]
    y_label, le = prepare_labels([train[1][i] for i in range(data_num)])

    tokenizer_train_text = text.Tokenizer()
    tokenizer_train_text.fit_on_texts(list(x_document))

    x_document_seq = tokenizer_train_text.texts_to_sequences(x_document)
    x_document_padded = sequence.pad_sequences(x_document_seq, maxlen=200)

    return x_document, x_document_padded, y_label, data_num
