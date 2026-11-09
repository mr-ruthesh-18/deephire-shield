"""DeepHire Shield - Step 1: collect, merge, clean and split the datasets.

Edit the DATASETS list below, then run:  python step1_prepare_data.py
Outputs (in ./data_ready): merged_jobs.csv, train.csv, test.csv, class_balance.png
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1) EDIT THIS: one entry per dataset file.
#    "columns" maps our standard names -> the column names in YOUR file.
#    Leave out any column your file does not have.
#    "label_col" is the fake/genuine column; "fraud_values" are the values that mean FRAUD.
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "emscad",
        "path": "fake_job_postings.csv",
        "columns": {
            "title": "title",
            "department": "department",
            "company_profile": "company_profile",
            "description": "description",
            "requirements": "requirements",
            "benefits": "benefits",
        },
        "label_col": "fraudulent",
        "fraud_values": [1],
    },
    # Pakistan job postings (Kaggle: zusmani/pakistans-job-market), Dec 2019 - Mar 2021.
    # NOTE: this file's "label" column is the listing type (Gallery/Hot/Premium Job),
    # NOT fake/genuine, so it is ignored. Following the base paper, all rows are genuine (0).
    {
        "name": "pakistan",
        "path": "pakistan_jobs.csv",
        "columns": {"title": "Job Name", "department": "Department", "description": "JD"},
        "label_col": None,
        "default_label": 0,
    },
    # US job postings (Kaggle: promptcloud/indeed-job-posting-dataset), Jul - Aug 2019.
    # The downloaded file may carry a different name (e.g. "marketing_sample_for_trulia_com...csv")
    # but it holds the Indeed job postings: 30,002 rows x 30 columns. All rows genuine (0).
    {
        "name": "us_jobs",
        "path": "us_jobs.csv",
        "columns": {
            "title": "Job Title",
            "company_profile": "Companydescription",
            "description": "Job Description",
        },
        "label_col": None,
        "default_label": 0,
    },
]

OUT_DIR = Path("data_ready")
TEST_SIZE = 0.20
SEED = 42
MIN_WORDS = 10
FIELDS = ["title", "department", "company_profile", "description", "requirements", "benefits"]


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))             # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)     # URLs
    text = re.sub(r"\S+@\S+\.\S+", " ", text)              # emails
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)           # HTML entities
    text = re.sub(r"[^\w\s.,;:!?$%&()'/+-]", " ", text)    # odd symbols
    return re.sub(r"\s+", " ", text).strip()


def load_one(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["path"])
    print(f"\n[{cfg['name']}] loaded {len(df)} rows; columns: {list(df.columns)}")
    parts = []
    for std in FIELDS:
        src = cfg["columns"].get(std)
        if src and src in df.columns:
            col = df[src].fillna("").astype(str).map(clean)
            parts.append(col.map(lambda v, s=std: f"{s.replace('_', ' ').title()}: {v}" if v else ""))
    if not parts:
        raise ValueError(f"No text columns found for {cfg['name']}; check 'columns' mapping.")
    text = pd.concat(parts, axis=1).apply(lambda r: " ".join(x for x in r if x), axis=1)
    if cfg.get("label_col"):
        label = df[cfg["label_col"]].isin(cfg["fraud_values"]).astype(int)
    else:  # dataset has no fake/genuine column: use the fixed label given below
        label = pd.Series(cfg["default_label"], index=df.index, dtype=int)
    out = pd.DataFrame({"job_text": text, "fraudulent": label, "source": cfg["name"]})
    print(f"[{cfg['name']}] fraud={label.sum()}  genuine={(label == 0).sum()}")
    return out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = pd.concat([load_one(c) for c in DATASETS], ignore_index=True)
    print(f"\nMerged rows: {len(data)}")

    # Remove empty / very short posts
    data = data[data["job_text"].str.split().str.len() >= MIN_WORDS]

    # Remove duplicates (same text). If the same text has conflicting labels, drop it entirely.
    data["key"] = data["job_text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    conflict = data.groupby("key")["fraudulent"].nunique()
    conflict_keys = conflict[conflict > 1].index
    data = data[~data["key"].isin(conflict_keys)]
    before = len(data)
    data = data.drop_duplicates(subset="key").drop(columns="key").reset_index(drop=True)
    print(f"Removed {before - len(data)} duplicate posts and {len(conflict_keys)} label-conflict texts")
    data.insert(0, "job_id", range(len(data)))

    # Stratified split AFTER de-duplication, so no text appears in both train and test
    train, test = train_test_split(data, test_size=TEST_SIZE, stratify=data["fraudulent"], random_state=SEED)
    assert not set(train["job_text"]) & set(test["job_text"]), "leakage between train and test!"

    data.to_csv(OUT_DIR / "merged_jobs.csv", index=False)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print("\nFinal class counts (0 = genuine, 1 = fraudulent):")
    for name, d in [("all", data), ("train", train), ("test", test)]:
        vc = d["fraudulent"].value_counts().to_dict()
        print(f"  {name:5s} total={len(d):6d}  genuine={vc.get(0, 0):6d}  fraud={vc.get(1, 0):5d}")
    print("\nBy source:\n", data.groupby(["source", "fraudulent"]).size().unstack(fill_value=0))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        counts = data["fraudulent"].value_counts().sort_index()
        plt.figure(figsize=(5, 4))
        bars = plt.bar(["Genuine", "Fraudulent"], [counts.get(0, 0), counts.get(1, 0)], color=["#548235", "#C00000"])
        plt.bar_label(bars)
        plt.title("Class distribution (after cleaning)")
        plt.ylabel("Number of job postings")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "class_balance.png", dpi=150)
        print("Saved class_balance.png")
    except ImportError:
        print("matplotlib not installed; skipped chart")


if __name__ == "__main__":
    main()
