import os
import re
from glob import glob

from src.Time_Series.config import cfg

# %% 2) Discover subjects
paths = sorted(glob(os.path.join(cfg.data_dir, "S*", "*.pkl"))) \
     or sorted(glob(os.path.join(cfg.data_dir, "S*.pkl")))
assert paths, f"No .pkl found under {cfg.data_dir} (expected S*/S*.pkl or S*.pkl)"


def subject_id_from_path(p: str) -> str:
    m = re.search(r"(S\d+)", p)
    return m.group(1) if m else os.path.basename(p)


subject_ids = [subject_id_from_path(p) for p in paths]
print("Found subjects:", subject_ids)
print("Num files:", len(paths))
