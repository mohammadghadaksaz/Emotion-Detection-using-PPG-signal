import pickle
import numpy as np
from typing import List, Tuple
from src.Time_Series.preprocessing import butter_bandpass, build_transition_mask, overlap_with_mask

def load_subject_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def majority_label_for_interval(lbl_700hz, fs_lbl, t0_s, t1_s, valid=(1,2,3,4)):
    i0, i1 = int(round(t0_s*fs_lbl)), int(round(t1_s*fs_lbl))
    i1 = min(i1, len(lbl_700hz))
    seg = lbl_700hz[i0:i1]
    seg = seg[np.isin(seg, valid)]
    if seg.size == 0: return -1
    vals, cnt = np.unique(seg, return_counts=True)
    return int(vals[np.argmax(cnt)])

def make_ppg_windows_for_subject(d: dict, cfg) -> Tuple[List[np.ndarray], List[int], int]:
    """
    Build (filtered) PPG windows + labels for one subject.
    Robust to very short signals that would otherwise break sosfiltfilt.
    """
    ppg   = d["signal"]["wrist"]["BVP"].astype(np.float32)
    fs_bvp = 64       # 64
    labels = d["label"].astype(int)                # ~700 Hz
    fs_lbl = 700       # 700

    # Optional: identify subject for clearer warnings (if present)
    subj = d.get("subject", "unknown")

    # ---- Guard: extremely short signals ----
    # Using the same threshold logic as preprocessing (padlen≈27 for your filter)
    if ppg.size == 0:
        print(f"[WARN] Skipping subject {subj}: empty PPG.")
        return [], [], fs_bvp
    if ppg.size <= 27:
        print(f"[WARN] Skipping subject {subj}: PPG too short ({ppg.size} samples ≤ 27).")
        return [], [], fs_bvp

    # ---- 1) Filter the entire sequence (robust function handles short arrays) ----
    lo, hi = cfg.ppg_band
    ppg_f = butter_bandpass(ppg, fs_bvp, lo, hi)

    # ---- 2) Build transition mask ----
    mask = build_transition_mask(labels, fs_lbl, cfg.transition_margin_s)

    # ---- 3) Windowing & labeling ----
    win  = int(cfg.win_s  * fs_bvp)
    step = int(cfg.step_s * fs_bvp)

    X, Y = [], []
    for s in range(0, len(ppg_f) - win + 1, step):
        t0, t1 = s/fs_bvp, (s+win)/fs_bvp
        if overlap_with_mask(t0, t1, mask, fs_lbl):
            continue
        lab = majority_label_for_interval(labels, fs_lbl, t0, t1, cfg.classes_kept)
        if lab == -1:
            continue
        X.append(ppg_f[s:s+win])
        Y.append(cfg.label_map4[lab])

    return X, Y, fs_bvp