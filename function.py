import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
tqdm.pandas()

from text_dataset import TextDataset
from train_model import train_model


def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder


def get_score(y_true, y_pred):
    print("acc : {}".format(accuracy_score(y_true, y_pred)))
    print("f1-score: {}".format(f1_score(y_true, y_pred, average='weighted')))


def run_model(index, model, train_index, val_index, x_torch, y_torch, device):
    full_dataset = TextDataset(X=x_torch, y=y_torch)
    train_dataset = torch.utils.data.Subset(full_dataset, train_index)
    valid_dataset = torch.utils.data.Subset(full_dataset, val_index)

    batch_size = 32

    valid_preds, training_losses, valid_losses = train_model(index, model, train_dataset, valid_dataset,
                                                             batch_size, device)
    return valid_preds, training_losses, valid_losses
