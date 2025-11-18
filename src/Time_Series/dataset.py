import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

class PPGWindowsDataset(Dataset):
    """
    Holds PPG windows (1-D arrays) and labels; z-score using TRAIN stats only.
    Ensures each sample is (1, T) for Conv1d and labels are torch.long in [0..3].
    """
    def __init__(self, X, Y, mean=None, std=None, fit_stats=False, eps=1e-8):
        self.X = X
        self.Y = Y
        self.eps = eps

        if fit_stats:
            allc = np.concatenate(self.X) if len(self.X) else np.array([0.0], dtype=np.float32)
            self.mean = float(allc.mean())
            self.std  = float(allc.std() + eps)
        else:
            self.mean = float(mean)
            self.std  = float(std)

    def __len__(self): return len(self.Y)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32)          # (T,)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.reshape(1, -1)                                   # (1, T)
        y = int(self.Y[idx])                                   # 0..3
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
