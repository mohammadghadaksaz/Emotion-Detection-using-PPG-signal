
# %% 1) Imports & configuration
import os, re, json, random, numpy as np
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from glob import glob
from collections import Counter
from dataclasses import dataclass
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.signal import butter, sosfiltfilt, sosfilt

import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # only touch CUDA if it is actually usable
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print("[seed] Skipping CUDA seeding:", e)


@dataclass
class CFG:
    data_dir: str = "/dataset"   # folder with S*/S*.pkl or S*.pkl
    win_s: int = 60
    step_s: int = 30
    ppg_band: tuple = (0.5, 8.0)  # Hz
    transition_margin_s: int = 30
    batch_size: int = 32
    epochs: int = 25
    lr: float = 0.001           # helps avoid flat-loss plateau
    weight_decay: float = 3e-5
    patience: int = 6
    num_classes: int = 3         # <-- set to 2 for Stress vs Non-Stress
    binary_mode: bool = True      # <-- convenience flag
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: int = 1              # 0 silent, 1 per-epoch, 2 per-batch
    overfit_tiny: bool = False    # debug switch


cfg = CFG()
set_seed(42)

print(cfg)
