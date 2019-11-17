import numpy as np
import datetime

import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from tqdm import tqdm
tqdm.pandas()

from ELMoForManyLangs.elmoformanylangs import Embedder
from ELMo import ELMoNet
from prepare_data import prepare_train, prepare_valid, prepare_test
from function import run_model, get_score
from combine_data import combine_train_valid, combine_train_test


def main():
    DATA_ROOT = './'
    word_model_path = "./word"
    char_model_path = "./letter"
    w2v_embedding_path = './entity_vector/entity_vector.model.bin'
    embedding_model = KeyedVectors.load_word2vec_format(w2v_embedding_path, binary=True)
    is_char = False

    x_train, x_train_padded, y_train, train_num = prepare_train(DATA_ROOT)
    x_valid, x_valid_padded, y_valid, valid_num = prepare_valid(DATA_ROOT, train_num)
    # x_test, x_test_padded, y_test, test_num = prepare_test(DATA_ROOT, train_num)

    x_padded, y_main, tokenizer_text, max_features = combine_train_valid(x_train, y_train, x_valid, y_valid)
    # x_padded, y_main, tokenizer_text, max_features = combine_train_test(x_train, y_train, x_test, y_test)

    word_index = tokenizer_text.word_index
    num_words = len(word_index)

    embedding_matrix = np.zeros((num_words + 1, 200))
    for word, i in tqdm(word_index.items()):
        if word in embedding_model.index2word:
            embedding_matrix[i] = embedding_model[word]

    print(embedding_matrix.shape)

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer_text.word_index.items()))


    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return words


    if is_char == True:
        char_e = Embedder(char_model_path)
        model = ELMoNet(char_e, embedding_matrix, 9, max_features, sequence_to_text)
        print('char mode')
    else:
        word_e = Embedder(word_model_path)
        model = ELMoNet(word_e, embedding_matrix, 9, max_features, sequence_to_text)
        print('word mode')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_torch = torch.tensor(x_padded, dtype=torch.long)
    y_torch = torch.tensor(y_main, dtype=torch.float32)

    train_index = [i for i in range(train_num)]
    val_index = [i for i in range(train_num, train_num + valid_num)]

    model = model.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net) # make parallel
        print("gpu parallel mode")
        cudnn.benchmark = True

    print(f"fold{0} start")
    valid_preds, training_losses, valid_losses = run_model(0, model, train_index, val_index, x_torch, y_torch, device)

    # テストデータの答え
    # test = pd.read_csv(Path(DATA_ROOT + 'livedoor') / 'test.csv', header=None)
    # y_true_test, le_true_test = prepare_labels([test[1][i] for i in range(test_num)])

    print(get_score(np.argmax(valid_preds, 1), np.argmax(y_valid, 1)))
    print(np.argmax(valid_preds, 1))

    date_now = datetime.datetime.now()
    plt.plot(training_losses)
    plt.plot(valid_losses)
    plt.savefig('./result/result' + date_now.strftime('_%Y%m%d_%H%M%S') + '.png')


if __name__ == '__main__':
    main()
