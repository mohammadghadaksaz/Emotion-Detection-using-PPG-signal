import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

# %% 6) Dataset & DataLoader
class PPGWindowsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32).reshape(1,-1)  # (1,T)
        y = int(self.Y[idx])                                         # 0..3
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def class_weights_from(y, n_classes):
    cnt = Counter(y)
    freqs = np.array([cnt.get(i, 0) for i in range(n_classes)], dtype=float)
    with np.errstate(divide='ignore'):
        inv = np.where(freqs > 0, 1.0/freqs, 0.0)
    w_sum = inv.sum()
    if w_sum > 0:
        inv *= (n_classes / w_sum)
    return torch.tensor(inv, dtype=torch.float)
