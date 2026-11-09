"""DeepHire Shield - Step 5: train the final baseline model once and save it for the demo app.

Uses the settings chosen on the validation set in step 4:
    learning rate 0.0003, dropout 0.5, 4 epochs, seed 42.
Needs step4_tune_and_evaluate.py in the same folder (it reuses the model code).

Output (./model_export): baseline.pt (weights) and scaler.npz (mean/std used to standardise features).
Run on Colab (GPU):   python step5_export_model.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from step4_tune_and_evaluate import DenseNet, metrics, predict, train_one_epoch

LR, DROPOUT, EPOCHS, BATCH, SEED = 3e-4, 0.5, 4, 32, 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("bert_features"))
    ap.add_argument("--out", type=Path, default=Path("model_export"))
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = np.load(a.dir / "train_sub_smobd.npz", allow_pickle=True)
    te = np.load(a.dir / "test_bert.npz", allow_pickle=True)
    Xtr, ytr = tr["features"].astype(np.float32), tr["labels"].astype(np.float32)
    Xte, yte = te["features"].astype(np.float32), te["labels"].astype(int)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd).to(device)
    ytr_t = torch.tensor(ytr).to(device)
    Xte_t = torch.tensor((Xte - mu) / sd).to(device)

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    model = DenseNet(DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(1, EPOCHS + 1):
        train_one_epoch(model, opt, loss_fn, Xtr_t, ytr_t, BATCH, gen, device)
        print(f"epoch {ep}/{EPOCHS} done", flush=True)

    m = metrics(yte, predict(model, Xte_t))
    print("Test check (one seed, so it will differ a little from the 5-seed average):")
    print({k: round(v, 4) for k, v in m.items() if k in ("recall", "precision", "f1", "balanced_acc", "g_mean")})

    torch.save(model.cpu().state_dict(), a.out / "baseline.pt")
    np.savez(a.out / "scaler.npz", mu=mu, sd=sd)
    print(f"Saved baseline.pt and scaler.npz to {a.out}/")


if __name__ == "__main__":
    main()
