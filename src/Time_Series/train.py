"""
LOSO split + training utilities:

- split_loso(...)
- build_fold_data(...)
- train_one_fold(...)
"""

import os
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.Time_Series.config import cfg
from src.Time_Series.subjects import subject_id_from_path
from src.Time_Series.windowing import make_ppg_windows_for_subject
from src.Time_Series.dataset import PPGWindowsDataset, class_weights_from
from src.Time_Series.models.cnn_lstm import ParallelCNN_LSTM
from src.Time_Series.metrics import f1_macro_np

import os

# %% 9) LOSO split + training utilities
def split_loso(subject_ids, test_sid, val_frac=0.2, seed=42):
    train_val = [s for s in subject_ids if s != test_sid]
    n_val = max(1, int(round(len(train_val)*val_frac)))
    rng = np.random.default_rng(seed + int(re.sub(r"\D","", test_sid) or 0))
    val = sorted(rng.choice(train_val, size=n_val, replace=False).tolist())
    train = sorted([s for s in train_val if s not in val])
    return train, val, [test_sid]


def build_fold_data(train_ids, val_ids, test_ids, paths, cfg):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = [], [], [], [], [], []
    for p in paths:
        sid = subject_id_from_path(p)
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        X, Y, _ = make_ppg_windows_for_subject(d, cfg)
        if   sid in train_ids:
            X_train += X; Y_train += Y
        elif sid in val_ids:
            X_val   += X; Y_val   += Y
        elif sid in test_ids:
            X_test  += X; Y_test  += Y
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)


def train_one_fold(X_train, Y_train, X_val, Y_val, X_test, Y_test, cfg):
    k = cfg.num_classes
    ds_train = PPGWindowsDataset(X_train, Y_train)
    ds_val   = PPGWindowsDataset(X_val,   Y_val)
    ds_test  = PPGWindowsDataset(X_test,  Y_test)

    print(f"[data] n_train={len(ds_train)}  n_val={len(ds_val)}  n_test={len(ds_test)}")
    if len(Y_train):
        print("[data] train class dist:", np.bincount(np.array(Y_train), minlength=k))
        all_train = np.concatenate([np.asarray(w, dtype=np.float32) for w in X_train]) if X_train else np.array([])
        if all_train.size:
            print(f"[norm] global train std: {float(all_train.std()):.6f}")
            if all_train.std() < 1e-6:
                print("[WARN] global train std ~0 (training may stall).")
    else:
        raise ValueError("No training windows after preprocessing.")

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False)

    model = ParallelCNN_LSTM(
        n_classes=k,       # k=4 for baseline/stress/amusement/meditation
        lstm_hidden=128,   # as in the paper-style variant
        lstm_layers=1,
        dropout=0.30
    ).to(cfg.device)

    cw = class_weights_from(Y_train, k).to(cfg.device)
    print("[train] class weights:", cw.detach().cpu().numpy().round(4).tolist())
    crit = nn.CrossEntropyLoss(weight=cw)
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.overfit_tiny and len(X_train) >= 64:
        print("[dbg] Overfit tiny mode ON")
        ds_small = PPGWindowsDataset(X_train[:64], Y_train[:64])
        train_loader = DataLoader(ds_small, batch_size=32, shuffle=True)
        val_loader   = DataLoader(ds_small, batch_size=32, shuffle=False)

    ckpt_dir = getattr(cfg, "ckpt_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_model_fold.h5")

    best_f1, best_state, wait = -1.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, n_batches = 0.0, 0

        iterator = train_loader
        if cfg.verbose >= 2:
            from tqdm import tqdm
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]", leave=False)

        for xb, yb in iterator:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()); n_batches += 1
            if cfg.verbose >= 2:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval(); all_p, all_t = [], []
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(cfg.device))
                val_loss_sum += float(crit(logits, yb.to(cfg.device)).item()); val_batches += 1
                all_p += logits.argmax(1).cpu().numpy().tolist()
                all_t += yb.cpu().numpy().tolist()
        val_f1, cm = f1_macro_np(np.array(all_p), np.array(all_t), k)
        val_loss = val_loss_sum / max(1, val_batches)
        train_loss = running_loss / max(1, n_batches)

        if cfg.verbose >= 1:
            print(f"[Epoch {epoch+1:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macroF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = {k2: v.cpu() for k2, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            if cfg.verbose >= 1:
                print(f"[Checkpoint] New best F1={best_f1:.4f}. Saved to {ckpt_path}")

    model.load_state_dict({k2: v.to(cfg.device) for k2, v in best_state.items()})
    model.eval(); all_p, all_t = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb.to(cfg.device)).argmax(1).cpu().numpy().tolist()
            all_p += p; all_t += yb.cpu().numpy().tolist()
    test_f1, cm = f1_macro_np(np.array(all_p), np.array(all_t), k)
    print(f"[Test] macro-F1={test_f1:.4f}\n[Test] Confusion matrix:\n{cm}")
    print(f"[Info] Best model for this fold saved at: {ckpt_path}")

    return dict(val_macro_f1=float(best_f1), test_macro_f1=float(test_f1), cm=cm.tolist())
