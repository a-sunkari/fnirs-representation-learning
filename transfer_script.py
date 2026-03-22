from pathlib import Path
import shutil
import re

SRC = Path("/home/asunkari/fnirs-representation-learning/mprs_od_cw_exploration")      # folder with SubjXX_no_hrf_surrogate_XX
DST = Path("/home/asunkari/fnirs-representation-learning/snirf_dataset_2")       # main dataset folder

for subj_dir in SRC.iterdir():
    if not subj_dir.is_dir():
        continue

    subj_name = subj_dir.name  # e.g., Subj86
    dst_subj_dir = DST / subj_name
    dst_subj_dir.mkdir(exist_ok=True)

    for f in subj_dir.glob("*.snirf"):
        match = re.match(r"(Subj\d+)_no_hrf_surrogate_(\d+)\.snirf", f.name)
        if not match:
            continue

        surrogate_id = match.group(2).zfill(2)

        new_name = f"resting_clean_surrogate_{surrogate_id}.snirf"
        dst_path = dst_subj_dir / new_name

        shutil.copy2(f, dst_path)
        print(f"Copied: {f.name} -> {dst_path}")