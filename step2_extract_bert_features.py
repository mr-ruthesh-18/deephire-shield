"""DeepHire Shield - Step 2: extract frozen BERT features.

Reads data_ready/train.csv and test.csv (from step 1) and writes
bert_features/train_bert.npz and test_bert.npz  (features, labels, source, job_id).

Run on a GPU machine (Google Colab T4 is fine):
    python step2_extract_bert_features.py
Quick test on a few rows:
    python step2_extract_bert_features.py --limit 64
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-uncased"


def extract(csv_path, out_path, tokenizer, model, device, batch_size, max_length, limit):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0).reset_index(drop=True)
    texts = df["job_text"].fillna("").astype(str).tolist()
    n = len(texts)
    # Sort by length so each batch has similar lengths (much faster), then restore the order.
    order = np.argsort([-len(t) for t in texts])
    feats = np.zeros((n, 768), dtype=np.float32)
    start_time = time.time()
    model.eval()
    with torch.inference_mode():
        for b, i in enumerate(range(0, n, batch_size)):
            idx = order[i:i + batch_size]
            tok = tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = model(**tok).last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean pooling -> 768 values
            feats[idx] = pooled.float().cpu().numpy()
            if b % 20 == 0 or i + batch_size >= n:
                done = min(i + batch_size, n)
                print(f"{csv_path.name}: {done}/{n} rows  ({time.time() - start_time:.0f}s)", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, features=feats, labels=df["fraudulent"].to_numpy(np.int64),
        source=df["source"].astype(str).to_numpy(), job_id=df["job_id"].to_numpy(np.int64))
    print(f"Saved {feats.shape} to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("data_ready"))
    ap.add_argument("--output-dir", type=Path, default=Path("bert_features"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--limit", type=int, default=0, help="only use N random rows per split (testing)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else "  (CPU is very slow - use a GPU)"))
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device)
    for split in ("train", "test"):
        out = a.output_dir / f"{split}_bert.npz"
        if out.exists() and not a.limit:
            print(f"{out} already exists - skipping (delete it to redo)")
            continue
        extract(a.input_dir / f"{split}.csv", out, tokenizer, model, device,
                a.batch_size, a.max_length, a.limit)


if __name__ == "__main__":
    main()
