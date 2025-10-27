import numpy as np
from scipy.signal import butter, sosfiltfilt

def butter_bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)

