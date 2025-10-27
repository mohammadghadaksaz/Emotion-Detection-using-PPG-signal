import numpy as np
from scipy.signal import butter, sosfiltfilt

def butter_bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)

def build_transition_mask(labels, fs_lbl, margin_s=30):
    change_idx = np.where(np.diff(labels) != 0)[0]
    mask = np.zeros_like(labels, dtype=bool)
    m = int(margin_s * fs_lbl)
    for i in change_idx:
        a, b = max(0, i - m), min(len(labels), i + m)
        mask[a:b] = True
    return mask

def overlap_with_mask(t0_s, t1_s, mask, fs_lbl):
    i0, i1 = int(t0_s * fs_lbl), int(t1_s * fs_lbl)
    i1 = min(i1, len(mask))
    return mask[i0:i1].any()