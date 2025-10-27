import os, re, random, numpy as np, torch
from collections import Counter


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def subject_id_from_path(p: str) -> str:
    # works for dataset/S2.pkl or dataset/S2/S2.pkl
    m = re.search(r"(S\d+)", p)
    return m.group(1) if m else os.path.basename(p)

def class_weights_from(y, n_classes):
    cnt = Counter(y); n = sum(cnt.values())
    w = [n/(n_classes*cnt[i]) if cnt.get(i, 0) > 0 else 0.0 for i in range(n_classes)]
    return torch.tensor(w, dtype=torch.float)