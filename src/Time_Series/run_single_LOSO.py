
import json
import os
import torch

from src.Time_Series.config import cfg
from src.Time_Series.subjects import subject_ids, paths
from src.Time_Series.train import split_loso, build_fold_data, train_one_fold

# %% 10) Run a single LOSO fold (example)
test_sid = subject_ids[0]
train_ids, val_ids, test_ids = split_loso(subject_ids, test_sid, val_frac=0.2, seed=42)
print(f"LOSO: TEST={test_sid} | TRAIN={len(train_ids)} subjects | VAL={len(val_ids)} subjects")

X_train, Y_train, X_val, Y_val, X_test, Y_test = build_fold_data(train_ids, val_ids, test_ids, paths, cfg)
stats = train_one_fold(X_train, Y_train, X_val, Y_val, X_test, Y_test, cfg)
print(json.dumps(stats, indent=2))

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("cfg.device:", cfg.device)
if cfg.device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True  # slight speedup for fixed-size inputs
