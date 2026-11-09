"""DeepHire Shield - Step 4 (v2): tune on a VALIDATION set, then evaluate once on the test set.

Training  : bert_features/train_sub_smobd.npz  (balanced with SMOBD-SMOTE)
Validation: bert_features/val_bert.npz         (held out before balancing, original class ratio)
Test      : bert_features/test_bert.npz        (untouched)

For each model it tries a small grid (learning rate x dropout), checks the validation balanced
accuracy after every epoch, and keeps the best setting. The chosen setting is then trained from
scratch with several seeds and scored on the test set. Test data is never used to choose anything.

Outputs (./results_v2): tuning_log.csv, best_settings.csv, results_per_seed.csv, summary.csv,
                        metric_comparison.png, confusion_matrices.png

Run:    python step4_tune_and_evaluate.py
Smoke test (fast, for checking that it runs):  python step4_tune_and_evaluate.py --quick
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DenseNet(nn.Module):
    """Baseline: 768 -> 128 -> 32 -> 1 feed-forward network."""
    def __init__(self, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 128), nn.ReLU(),
                                 nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNN2D(nn.Module):
    """Extension: 768 values reshaped to a 24 x 32 grid, then two Conv2D + MaxPool blocks."""
    def __init__(self, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * 6 * 8, 128),
                                nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.fc(self.conv(x.view(-1, 1, 24, 32))).squeeze(1)


MODELS = {"Neural Network (baseline)": DenseNet, "CNN2D (extension)": CNN2D}


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "specificity": spec,
            "f1": f1, "balanced_acc": (rec + spec) / 2, "g_mean": float(np.sqrt(rec * spec)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def train_one_epoch(model, opt, loss_fn, X, y, batch_size, gen, device):
    model.train()
    perm = torch.randperm(len(X), generator=gen)
    for i in range(0, len(X), batch_size):
        idx = perm[i:i + batch_size].to(device)
        opt.zero_grad()
        loss_fn(model(X[idx]), y[idx]).backward()
        opt.step()


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = [(torch.sigmoid(model(X[i:i + 2048])) >= 0.5).long().cpu() for i in range(0, len(X), 2048)]
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("bert_features"))
    ap.add_argument("--out", type=Path, default=Path("results_v2"))
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true", help="tiny grid, 3 epochs, 2 seeds (smoke test)")
    a = ap.parse_args()
    lrs, drops = [1e-3, 3e-4], [0.1, 0.3, 0.5]
    if a.quick:
        lrs, drops, a.max_epochs, a.seeds = [1e-3], [0.1, 0.5], 3, a.seeds[:2]
    a.out.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tr = np.load(a.dir / "train_sub_smobd.npz", allow_pickle=True)
    va = np.load(a.dir / "val_bert.npz", allow_pickle=True)
    te = np.load(a.dir / "test_bert.npz", allow_pickle=True)
    Xtr, ytr = tr["features"].astype(np.float32), tr["labels"].astype(np.float32)
    Xva, yva = va["features"].astype(np.float32), va["labels"].astype(int)
    Xte, yte, src = te["features"].astype(np.float32), te["labels"].astype(int), te["source"].astype(str)
    print(f"Train {Xtr.shape} (fraud={int(ytr.sum())})  Val {Xva.shape} (fraud={int(yva.sum())})  "
          f"Test {Xte.shape} (fraud={int(yte.sum())})")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6          # training statistics only
    to_t = lambda arr: torch.tensor((arr - mu) / sd).to(device)
    Xtr_t, Xva_t, Xte_t = to_t(Xtr), to_t(Xva), to_t(Xte)
    ytr_t = torch.tensor(ytr).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    # ---------------- 1) tune on validation ----------------
    log, best = [], {}
    for name, cls in MODELS.items():
        print(f"\n=== Tuning {name} ===")
        for lr in lrs:
            for dr in drops:
                torch.manual_seed(0)
                gen = torch.Generator().manual_seed(0)
                model = cls(dr).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                for ep in range(1, a.max_epochs + 1):
                    train_one_epoch(model, opt, loss_fn, Xtr_t, ytr_t, a.batch_size, gen, device)
                    m = metrics(yva, predict(model, Xva_t))
                    log.append({"model": name, "lr": lr, "dropout": dr, "epoch": ep,
                                "val_balanced_acc": m["balanced_acc"], "val_recall": m["recall"],
                                "val_precision": m["precision"], "val_f1": m["f1"]})
                    if name not in best or m["balanced_acc"] > best[name]["val_balanced_acc"]:
                        best[name] = {"lr": lr, "dropout": dr, "epochs": ep, "val_balanced_acc": m["balanced_acc"]}
                print(f"  lr={lr} dropout={dr}: best val balanced acc so far = {best[name]['val_balanced_acc']:.4f}",
                      flush=True)
        print(f"  -> chosen for {name}: {best[name]}")
    pd.DataFrame(log).to_csv(a.out / "tuning_log.csv", index=False)
    pd.DataFrame([{"model": k, **v} for k, v in best.items()]).to_csv(a.out / "best_settings.csv", index=False)

    # ---------------- 2) final runs on the test set ----------------
    subsets = {"full_test": np.ones(len(yte), bool), "emscad_only": src == "emscad"}
    rows, cms = [], {}
    for name, cls in MODELS.items():
        b = best[name]
        for seed in a.seeds:
            print(f"Final run: {name}, seed {seed}", flush=True)
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = cls(b["dropout"]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=b["lr"])
            for _ in range(b["epochs"]):
                train_one_epoch(model, opt, loss_fn, Xtr_t, ytr_t, a.batch_size, gen, device)
            pred = predict(model, Xte_t)
            for sub, mask in subsets.items():
                rows.append({"model": name, "seed": seed, "test_set": sub, **metrics(yte[mask], pred[mask])})
            if seed == a.seeds[0]:
                cms[name] = confusion_matrix(yte, pred, labels=[0, 1])
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "results_per_seed.csv", index=False)

    cols = ["recall", "precision", "specificity", "f1", "balanced_acc", "g_mean", "accuracy"]
    summ = res.groupby(["model", "test_set"])[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(a.out / "summary.csv")
    print("\nChosen settings (from validation):")
    print(pd.DataFrame([{"model": k, **v} for k, v in best.items()]).round(4).to_string(index=False))
    print(f"\nTest results, mean and std over seeds {a.seeds}:")
    print(summ[[(c, s) for c in ["recall", "precision", "f1", "balanced_acc", "g_mean"] for s in ("mean", "std")]].to_string())

    # ---------------- 3) plots ----------------
    keys = ["recall", "precision", "f1", "balanced_acc", "g_mean"]
    x, w = np.arange(len(keys)), 0.38
    plt.figure(figsize=(8, 4))
    for k, (name, color) in enumerate(zip(MODELS, ["#8EA9DB", "#2F5496"])):
        sub = res[(res.model == name) & (res.test_set == "full_test")][keys]
        plt.bar(x + (k - 0.5) * w, sub.mean(), w, yerr=sub.std(), capsize=3, label=name, color=color)
    plt.xticks(x, keys); plt.ylim(0, 1.05); plt.legend(fontsize=8)
    plt.title(f"Test results, mean +/- std over {len(a.seeds)} seeds (full test set)")
    plt.tight_layout(); plt.savefig(a.out / "metric_comparison.png", dpi=150); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, cm) in zip(axs, cms.items()):
        ax.imshow(cm, cmap="Blues"); ax.set_title(f"{name}, seed {a.seeds[0]}", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Genuine", "Fraud"]); ax.set_yticklabels(["Genuine", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(a.out / "confusion_matrices.png", dpi=150); plt.close()
    print(f"\nSaved everything to {a.out}/")


if __name__ == "__main__":
    main()
