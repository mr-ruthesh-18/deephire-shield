"""DeepHire Shield - Step 3 (v2): hold out a validation set, then balance ONLY the rest with SMOBD-SMOTE.

Input : bert_features/train_bert.npz            (from step 2)
Output: bert_features/val_bert.npz              (15% of train, original class ratio, no synthetic rows)
        bert_features/train_sub_smobd.npz       (other 85%, balanced with SMOBD-SMOTE)
The test set is never touched.

Install once:  pip install smote-variants
Run:           python step3_split_and_smobd.py
"""
import argparse
from pathlib import Path

import numpy as np
import smote_variants as sv
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("bert_features"))
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = np.load(a.dir / "train_bert.npz", allow_pickle=True)
    X, y, src = d["features"].astype(np.float32), d["labels"].astype(int), d["source"].astype(str)
    idx_tr, idx_val = train_test_split(np.arange(len(y)), test_size=a.val_size, stratify=y, random_state=a.seed)

    np.savez_compressed(a.dir / "val_bert.npz", features=X[idx_val], labels=y[idx_val], source=src[idx_val])
    print(f"Validation set: {len(idx_val)} rows, fraud={int(y[idx_val].sum())}  (original ratio, saved)")

    Xt, yt = X[idx_tr], y[idx_tr]
    print(f"Before SMOBD: {Xt.shape}  genuine={int((yt == 0).sum())}  fraud={int((yt == 1).sum())}")
    # max_eps must be large for 768-dim vectors; with the default (1.0) SMOBD silently adds nothing.
    Xr, yr = sv.SMOBD(proportion=1.0, max_eps=np.inf, random_state=a.seed).sample(Xt, yt)
    Xr, yr = np.asarray(Xr, dtype=np.float32), np.asarray(yr, dtype=np.int64)
    print(f"After SMOBD : {Xr.shape}  genuine={int((yr == 0).sum())}  fraud={int((yr == 1).sum())}")
    if len(yr) == len(yt):
        raise SystemExit("SMOBD did not add any samples - do not use this output.")

    np.savez_compressed(a.dir / "train_sub_smobd.npz", features=Xr, labels=yr)
    print("Saved train_sub_smobd.npz and val_bert.npz")


if __name__ == "__main__":
    main()
