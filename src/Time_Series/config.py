import os, random, numpy as np, torch
from dataclasses import dataclass

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print("[seed] Skipping CUDA seeding:", e)

@dataclass
class CFG:
    data_dir: str = "./dataset"
    win_s: int = 60
    step_s: int = 30
    ppg_band: tuple = (0.5, 8.0)
    transition_margin_s: int = 30
    batch_size: int = 32
    epochs: int = 40
    lr: float = 0.001
    weight_decay: float = 3e-5
    patience: int = 6
    num_classes: int = 4
    binary_mode: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: int = 1
    overfit_tiny: bool = False
