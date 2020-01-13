import pandas as pd

import numpy as np
from pathlib import Path
from janome.tokenizer import Tokenizer
from keras.preprocessing import text, sequence

from function import prepare_labels
from format_text import preprocess


def prepare_train(file_path):
    # トレインデータの用意
    train = pd.read_csv(file_path, header=None)
    train_num = train.shape[0]
    # train_num = 1000
    train_path = [train[0][i] for i in range(train_num)]
    train_document = [open(file_path + 'livedoor/' + train_path[j], 'r', encoding='utf-8').read() for j in
                      range(train_num)]
    # train_text = pd.Series(train_document)

    j_t = Tokenizer()

    x_train = []
    for train_text in train_document:
        x_train.append(preprocess(train_text, j_t))
    x_train = pd.Series(x_train)

    # x_train = preprocess(train_text)
    y_train, le = prepare_labels([train[1][i] for i in range(train_num)])

    tokenizer_train_text = text.Tokenizer()
    tokenizer_train_text.fit_on_texts(list(x_train))

    print(y_train)
    print(x_train)
    print(type(tokenizer_train_text))

    x_train_seq = tokenizer_train_text.texts_to_sequences(x_train)
    x_train_padded = sequence.pad_sequences(x_train_seq, maxlen=200)

    return x_train, x_train_padded, y_train, train_num


def prepare_valid(file_path, train_num):
    # validデータの用意
    valid = pd.read_csv(file_path, header=None)
    valid_num = valid.shape[0]
    # valid_num = 100
    valid_path = [valid[0][i] for i in range(valid_num)]
    valid_document = [open(file_path + 'livedoor/' + valid_path[j], 'r', encoding='utf-8').read() for j in
                      range(valid_num)]
    # valid_text = pd.Series(valid_document)

    j_t = Tokenizer()

    # x_valid = preprocess(valid_text, j_t)
    x_valid = []
    for valid_text in valid_document:
        x_valid.append(preprocess(valid_text, j_t))
    x_valid = pd.Series(x_valid)

    x_valid.index = [i for i in range(train_num, train_num + valid_num)]
    y_valid, le_valid = prepare_labels([valid[1][i] for i in range(valid_num)])

    print(y_valid)
    print(x_valid)

    tokenizer_valid_text = text.Tokenizer()
    tokenizer_valid_text.fit_on_texts(list(x_valid))

    x_valid_seq = tokenizer_valid_text.texts_to_sequences(x_valid)
    x_valid_padded = sequence.pad_sequences(x_valid_seq, maxlen=200)

    return x_valid, x_valid_padded, y_valid, valid_num


def prepare_test(file_path, train_num):
    # テストデータの用意
    test = pd.read_csv(file_path, header=None)
    # test_num = test.shape[0]
    test_num = 100
    test_path = [test[0][i] for i in range(test_num)]
    test_document = [open(file_path + 'livedoor/' + test_path[j], 'r', encoding='utf-8').read() for j in
                     range(test_num)]
    # test_text = pd.Series(test_document)

    j_t = Tokenizer()

    # x_test = preprocess(test_text, j_t)

    x_test = []
    for test_text in test_document:
        x_test.append(preprocess(test_text, j_t))
    x_test = pd.Series(x_test)

    x_test.index = [i for i in range(train_num, train_num + test_num)]
    y_test = np.eye(test_num, 9, 1)

    print(y_test.shape)
    print(x_test)

    tokenizer_text = text.Tokenizer()
    tokenizer_text.fit_on_texts(list(x_test))

    x_test_seq = tokenizer_text.texts_to_sequences(x_test)
    x_test_padded = sequence.pad_sequences(x_test_seq, maxlen=200)

    return x_test, x_test_padded, y_test, test_num
