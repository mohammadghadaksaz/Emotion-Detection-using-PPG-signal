import numpy as np
import pickle
import matplotlib.pyplot as plt

from src.Time_Series.config import cfg
from src.Time_Series.subjects import paths, subject_id_from_path
from src.Time_Series.preprocessing import butter_bandpass

# %% 4) Quick look: raw vs filtered (first subject)
demo_path = paths[0]
with open(demo_path, "rb") as f:
    d0 = pickle.load(f, encoding="latin1")

ppg = d0["signal"]["wrist"]["BVP"].astype(np.float32)
fs_bvp = 64
labels = d0["label"].astype(int)
fs_lbl = 700
print("Demo subject:", subject_id_from_path(d0.get("subject","") or demo_path))
print("PPG len:", len(ppg), "fs_bvp:", fs_bvp, "| labels fs:", fs_lbl)

lo, hi = cfg.ppg_band
print(ppg.shape)
ppg_f = butter_bandpass(ppg[:, 0], fs_bvp, lo, hi)

sec = 30
n = min(len(ppg), fs_bvp*sec)
plt.figure()
plt.plot(np.arange(n)/fs_bvp, ppg[:n], label="raw")
plt.plot(np.arange(n)/fs_bvp, ppg_f[:n], label="filtered", alpha=0.85)
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.title("PPG: raw vs filtered (first 30s)")
plt.legend() 
plt.savefig("figs/quickview.pdf", bbox_inches="tight")
plt.show()
