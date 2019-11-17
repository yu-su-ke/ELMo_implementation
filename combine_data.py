import numpy as np
import pandas as pd

from keras.preprocessing import text, sequence


def combine_train_valid(x_train, y_train, x_valid, y_valid):
    all_text = pd.concat([x_train, x_valid])
    tokenizer_text = text.Tokenizer()
    tokenizer_text.fit_on_texts(list(all_text))

    x_seq = tokenizer_text.texts_to_sequences(all_text)
    x_padded = sequence.pad_sequences(x_seq, maxlen=200)

    max_features = None
    max_features = max_features or len(tokenizer_text.word_index) + 1
    print(max_features)

    y_main = np.concatenate([y_train, y_valid])

    return x_padded, y_main, tokenizer_text, max_features


def combine_train_test(x_train, y_train, x_test, y_test):
    all_text = pd.concat([x_train, x_test])
    tokenizer_text = text.Tokenizer()
    tokenizer_text.fit_on_texts(list(all_text))

    x_seq = tokenizer_text.texts_to_sequences(all_text)
    x_padded = sequence.pad_sequences(x_seq, maxlen=200)

    max_features = None
    max_features = max_features or len(tokenizer_text.word_index) + 1
    print(max_features)

    y_main = np.concatenate([y_train, y_test])

    return x_padded, y_main, tokenizer_text, max_features
