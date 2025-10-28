import numpy as np

def confusion_matrix(pred, true, k):
    m = np.zeros((k,k), dtype=int)
    for p,t in zip(pred, true): m[t, p] += 1
    return m

def f1_macro(pred, true, k):
    cm = confusion_matrix(pred, true, k)
    f1s = []
    for c in range(k):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2*prec*rec / (prec+rec+1e-12)
        f1s.append(f1)
    return float(np.mean(f1s)), cm
