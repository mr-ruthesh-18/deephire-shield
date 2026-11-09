"""DeepHire Shield - demo app.

Paste a job posting, get "Genuine" or "Fraudulent".
Flow: job text -> BERT features -> trained classifier -> verdict.
(SMOBD-SMOTE is used only while training, so it is not part of this app.)

Run:  streamlit run app.py        (needs the model_export folder next to this file)
"""
import re
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "model_export"
THRESHOLD = 0.5


class DenseNet(nn.Module):
    """Same layers as in step4_tune_and_evaluate.py (the baseline classifier)."""
    def __init__(self, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 128), nn.ReLU(),
                                 nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def clean(text: str) -> str:
    """Same cleaning as step1_prepare_data.py."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    text = re.sub(r"[^\w\s.,;:!?$%&()'/+-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Loading BERT and the classifier (first time only)...")
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    clf = DenseNet()
    clf.load_state_dict(torch.load(MODEL_DIR / "baseline.pt", map_location="cpu"))
    clf = clf.to(device).eval()
    sc = np.load(MODEL_DIR / "scaler.npz")
    return device, tok, bert, clf, sc["mu"], sc["sd"]


def fraud_probability(text: str) -> float:
    device, tok, bert, clf, mu, sd = load_resources()
    with torch.inference_mode():
        enc = tok([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = bert(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        vec = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
        x = torch.tensor((vec - mu) / sd, dtype=torch.float32).to(device)
        return float(torch.sigmoid(clf(x)).item())


# ------------------------------- page -------------------------------
st.set_page_config(page_title="DeepHire Shield", page_icon="🛡️")
st.title("🛡️ DeepHire Shield")
st.caption("Online recruitment fraud detection using BERT, SMOBD-SMOTE and a neural network classifier")

if not (MODEL_DIR / "baseline.pt").exists():
    st.error("The model_export folder (baseline.pt and scaler.npz) was not found next to app.py. "
             "Create it by running step5_export_model.py, then copy the folder here.")
    st.stop()

title = st.text_input("Job title (optional)")
company = st.text_area("Company description (optional)", height=80)
desc = st.text_area("Job description", height=220, placeholder="Paste the job posting text here...")

if st.button("Check this job posting", type="primary"):
    if len(desc.split()) < 10:
        st.warning("Please paste a longer job description (at least 10 words). The model was trained on full postings.")
    else:
        parts = []
        if clean(title):
            parts.append(f"Title: {clean(title)}")
        if clean(company):
            parts.append(f"Company Profile: {clean(company)}")
        parts.append(f"Description: {clean(desc)}")
        with st.spinner("Analysing..."):
            p = fraud_probability(" ".join(parts))
        if p >= THRESHOLD:
            st.error(f"🚨 Fraudulent (fraud score {p:.0%})")
        else:
            st.success(f"✅ Genuine (fraud score {p:.0%})")
        st.progress(min(max(p, 0.0), 1.0), text="Fraud score")
        st.info("This is a student research model. On our test data it caught about 82% of fake postings, "
                "but only about half of the postings it flagged were really fake. Treat the result as a warning "
                "sign, not as proof, and always check the company yourself before sharing personal or payment details.")

with st.expander("How it works"):
    st.markdown(
        "1. The job text is cleaned (HTML, links and emails removed).\n"
        "2. **BERT** turns the text into a 768-number vector.\n"
        "3. The trained **neural network** gives a fraud score between 0 and 1.\n"
        "4. A score of 0.5 or more is shown as **Fraudulent**, otherwise **Genuine**.\n\n"
        "**SMOBD-SMOTE** was used only during training, to balance the genuine and fraudulent examples."
    )
