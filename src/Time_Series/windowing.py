import numpy as np
import pickle
from collections import Counter

from src.Time_Series.config import cfg
from src.Time_Series.preprocessing import (
    butter_bandpass,
    build_transition_mask,
    overlap_with_mask,
    majority_label_for_interval,
)

# %% 5) Windowing (per-subject z-score; drop bad windows)
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def make_ppg_windows_for_subject(d, cfg, keep_classes=(1,2,3,4)):
    ppg    = d["signal"]["wrist"]["BVP"].astype(np.float32)
    fs_bvp = 64         # 64
    labels = d["label"].astype(int)                # ~700
    fs_lbl = 700        # 700
    subj   = d.get("subject", "unknown")

    if ppg.size == 0:
        print(f"[WARN] Skipping subject {subj}: empty PPG.")
        return [], [], fs_bvp

    # 1) Band-pass
    lo, hi = cfg.ppg_band
    ppg_f = butter_bandpass(ppg[:, 0], fs_bvp, lo, hi)
    if not np.isfinite(ppg_f).all():
        ppg_f = np.nan_to_num(ppg_f, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Per-subject z-score
    mu, sd = float(ppg_f.mean()), float(ppg_f.std())
    if not np.isfinite(sd) or sd < 1e-8:
        print(f"[WARN] Skipping subject {subj}: near-constant post-filter (std={sd:.3e}).")
        return [], [], fs_bvp
    ppg_f = (ppg_f - mu) / sd

    # 3) Transition mask
    mask = build_transition_mask(labels, fs_lbl, cfg.transition_margin_s)

    # 4) Windowing + labeling (drop NaN/constant windows)
    win  = int(cfg.win_s  * fs_bvp)
    step = int(cfg.step_s * fs_bvp)
    X, Y = [], []
    kept, dropped_const, dropped_nan = 0, 0, 0
    label_map4 = {1:0, 2:1, 3:2, 4:2}

    for s in range(0, len(ppg_f) - win + 1, step):
        t0, t1 = s/fs_bvp, (s+win)/fs_bvp
        if overlap_with_mask(t0, t1, mask, fs_lbl):  # skip transition overlaps
            continue
        lab = majority_label_for_interval(labels, fs_lbl, t0, t1, keep_classes)
        if lab == -1:
            continue
        if lab == 3:
            continue
        w = ppg_f[s:s+win]
        if not np.isfinite(w).all():
            dropped_nan += 1
            continue
        if np.std(w) < 1e-8:
            dropped_const += 1
            continue
        X.append(w.copy()); Y.append(label_map4[lab]); kept += 1

    if dropped_const or dropped_nan:
        print(f"[info] Subject {subj}: kept={kept}, dropped_const={dropped_const}, dropped_nan={dropped_nan}")
    return X, Y, fs_bvp
