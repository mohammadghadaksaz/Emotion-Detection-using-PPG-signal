from scipy.signal import butter, sosfiltfilt, sosfilt
import numpy as np

# %% 3) Preprocessing helpers (filter, masks, labeling)
def butter_bandpass(x, fs, lo, hi, order=4):
    """Robust zero-phase band-pass; falls back for short arrays."""
    x = np.asarray(x)
    sos = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band", output="sos")
    padlen = 3 * (sos.shape[0] - 1) + 1  # ~27 for order=4 bandpass
    n = x.size
    if n == 0:
        return x.astype(np.float32, copy=False)
    if n <= padlen:
        y = sosfilt(sos, x)
        y = sosfilt(sos, y[::-1])[::-1]
        return y.astype(np.float32, copy=False)
    try:
        y = sosfiltfilt(sos, x)
    except ValueError:
        y = sosfiltfilt(sos, x, padlen=0)
    return y.astype(np.float32, copy=False)


def build_transition_mask(labels, fs_lbl, margin_s=30):
    change_idx = np.where(np.diff(labels) != 0)[0]
    mask = np.zeros_like(labels, dtype=bool)
    m = int(margin_s * fs_lbl)
    for i in change_idx:
        a, b = max(0, i - m), min(len(labels), i + m)
        mask[a:b] = True
    return mask


def overlap_with_mask(t0_s, t1_s, mask, fs_lbl):
    i0, i1 = int(t0_s * fs_lbl), min(int(t1_s * fs_lbl), len(mask))
    return mask[i0:i1].any()


def majority_label_for_interval(lbl_700hz, fs_lbl, t0_s, t1_s, valid=(1,2,3,4)):
    i0, i1 = int(round(t0_s*fs_lbl)), min(int(round(t1_s*fs_lbl)), len(lbl_700hz))
    seg = lbl_700hz[i0:i1]
    seg = seg[np.isin(seg, valid)]
    if seg.size == 0:
        return -1
    vals, cnt = np.unique(seg, return_counts=True)
    return int(vals[np.argmax(cnt)])
