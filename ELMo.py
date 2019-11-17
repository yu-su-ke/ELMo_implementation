import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
tqdm.pandas()


class ELMoNet(nn.Module):
    def __init__(self, elmo_model, embedding_matrix, OUTPUT_DIM, max_features, sequence_to_text):
        super(ELMoNet, self).__init__()

        embed_size = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout(0.1)

        self.LSTM_UNITS = 128
        self.DENSE_HIDDEN_UNITS = self.LSTM_UNITS * 4

        self.elmo_embedder = elmo_model

        self.lstm1 = nn.LSTM(1024 + 200, self.LSTM_UNITS, bidirectional=True, batch_first=True)
        # self.lstm1 = nn.LSTM(200, self.LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(self.LSTM_UNITS * 2, self.LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(self.DENSE_HIDDEN_UNITS, self.DENSE_HIDDEN_UNITS)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(self.DENSE_HIDDEN_UNITS, OUTPUT_DIM)

        self.sequence_to_text = sequence_to_text


    def forward(self, x):
        l = x.shape[1]

        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        # 0 padding を除く
        x_p = [[i for i in y if i != 0] for y in x.cpu().detach().numpy()]

        # x_p : index, sentences : text
        sentences = list(map(self.sequence_to_text, x_p))

        h_elmo = self.elmo_embedder.sents2elmo(sentences)
        h_elmo = [np.concatenate(
            [i, [[0] * 1024] * (l - len(i))], axis=0) if len(i) != l else i for i in h_elmo]
        h_elmo = torch.tensor(h_elmo).float().cuda()

        # fasttext vector と elmo vector を concat する.
        h_embcat = torch.cat([h_elmo, h_embedding], 2)

        # h_embcat = h_embedding

        h_lstm1, _ = self.lstm1(h_embcat)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        out = F.relu(self.linear1(h_conc))
        out = self.linear2(self.dropout(out))

        return F.log_softmax(out)
