import numpy as np

# %% 8) Metrics (macro-F1)
def confusion_matrix_np(pred, true, k):
    m = np.zeros((k,k), dtype=int)
    for p, t in zip(pred, true):
        m[t, p] += 1
    return m


def f1_macro_np(pred, true, k):
    cm = confusion_matrix_np(pred, true, k)
    f1s = []
    for c in range(k):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1s.append(2*prec*rec / (prec+rec+1e-12))
    return float(np.mean(f1s)), cm
