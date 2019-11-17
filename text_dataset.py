from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return [self.X[idx], self.y[idx]]
        return self.X[idx]
