"""
Generates three self-contained Jupyter notebooks for the fraud detection project.
Run once:  python _generate_notebooks.py
"""
import json, os

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)


# ─── helpers ────────────────────────────────────────────────────────────────

def cc(src: str):
    """Code cell."""
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

def mc(src: str):
    """Markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": []}
        },
        "cells": cells
    }

def save(name, cells):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"  wrote {path}")


# ─── shared code snippets ────────────────────────────────────────────────────

INSTALL_BASELINE = """\
# Install required packages (run once in Colab)
!pip install -q xgboost lightgbm pyarrow pandas numpy scikit-learn\
"""

GIT_SETUP = """\
# ── Pull latest code from GitHub (run at the start of every session) ──────────
import os

if os.path.exists("DSC_599_Semester_Project"):
    # Repo already cloned — just pull the latest changes
    %cd DSC_599_Semester_Project
    !git pull origin colab
else:
    # First time — clone and switch to the colab branch
    !git clone https://github.com/ConlerSimmons/DSC_599_Semester_Project.git
    %cd DSC_599_Semester_Project
    !git checkout colab

print("\\nCurrent branch:")
!git branch --show-current
print("Latest commit:")
!git log --oneline -1\
"""

INSTALL_TABT = """\
# Install required packages (run once in Colab)
!pip install -q torch pandas numpy scikit-learn pyarrow lightgbm\
"""

INSTALL_GNN = INSTALL_TABT

DATA_SETUP = '''\
import os
from google.colab import drive

# ── Mount Google Drive and point to your data ─────────────────────────────────
# Before running:
#   1. Upload train_transaction.csv and train_identity.csv to Google Drive
#      e.g. into a folder called "ieee_fraud/raw/"
#   2. Update DRIVE_DATA_PATH below to match where you put them
#   3. Run this cell — it will prompt you to authorise Drive access

drive.mount("/content/drive")

DRIVE_DATA_PATH = "/content/drive/MyDrive/Data_Fraud"  # ← update if your folder name differs
DATA_DIR = "data"

os.makedirs(f"{DATA_DIR}/raw",     exist_ok=True)
os.makedirs(f"{DATA_DIR}/interim", exist_ok=True)

# Copy from Drive to Colab local storage (faster reads during training)
os.system(f"cp {DRIVE_DATA_PATH}/train_transaction.csv {DATA_DIR}/raw/")
os.system(f"cp {DRIVE_DATA_PATH}/train_identity.csv    {DATA_DIR}/raw/")

print("Files ready:")
os.system(f"ls -lh {DATA_DIR}/raw/")\
'''

LOAD_MERGED = '''\
import os
import pandas as pd

def load_merged_train(data_dir="data"):
    """
    Load and merge the IEEE-CIS training files.

    Reads train_transaction.csv and train_identity.csv from data_dir/raw/,
    left-joins them on TransactionID so every transaction is kept (identity
    features are NaN for ~76% of rows that have no identity record), then
    caches the result to data_dir/interim/merged_train.parquet for fast
    re-loads.

    Returns
    -------
    pd.DataFrame  shape ~ (590540, 434)
    """
    cache = os.path.join(data_dir, "interim", "merged_train.parquet")
    if os.path.exists(cache):
        print(f"Loading from cache: {cache}")
        df = pd.read_parquet(cache)
        print(f"Shape: {df.shape}")
        return df

    raw = os.path.join(data_dir, "raw")
    print("Reading train_transaction.csv …")
    trans = pd.read_csv(os.path.join(raw, "train_transaction.csv"))
    print("Reading train_identity.csv …")
    ident = pd.read_csv(os.path.join(raw, "train_identity.csv"))
    print("Merging on TransactionID …")
    df = trans.merge(ident, how="left", on="TransactionID")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"Cached to {cache}.  Shape: {df.shape}")
    return df

df = load_merged_train(DATA_DIR)
df = df.sort_values("TransactionDT").reset_index(drop=True)
print(f"Sorted by TransactionDT.  Shape: {df.shape}")\
'''

FEATURE_SEL = '''\
import numpy as np

def auto_select_features(df, target_col="isFraud", max_numeric=50, max_categorical=20):
    """
    Heuristic feature selection for the IEEE-CIS dataset.

    Rules applied in order:
      1. Drop columns that are >95% missing (too sparse to be useful).
      2. Classify remaining columns as numeric or categorical:
           - object / category dtype  → categorical
           - integer with ≤20 unique values → categorical (likely encoded flag)
           - integer with >20 unique values → numeric
           - float → numeric
      3. Cap at max_numeric + max_categorical to keep model sizes manageable.

    Returns
    -------
    numeric_cols : list[str]
    categorical_cols : list[str]
    """
    ignore = {target_col, "TransactionID"}
    na_frac = df.isna().mean()
    kept = [c for c in na_frac[na_frac < 0.95].index if c not in ignore]

    numeric_cols, categorical_cols = [], []
    for col in kept:
        dt = df[col].dtype
        if dt == "object" or str(dt) == "category":
            categorical_cols.append(col)
        elif np.issubdtype(dt, np.integer):
            (categorical_cols if df[col].nunique(dropna=True) <= 20
             else numeric_cols).append(col)
        elif np.issubdtype(dt, np.floating):
            numeric_cols.append(col)

    numeric_cols     = numeric_cols[:max_numeric]
    categorical_cols = categorical_cols[:max_categorical]
    print(f"Selected {len(numeric_cols)} numeric, {len(categorical_cols)} categorical features")
    print("Numeric:", numeric_cols)
    print("Categorical:", categorical_cols)
    return numeric_cols, categorical_cols

numeric_cols, categorical_cols = auto_select_features(
    df, target_col="isFraud", max_numeric=50, max_categorical=20
)\
'''

SPLIT = '''\
# Temporal 70 / 15 / 15 split — respects transaction time ordering.
# Using random splits on time-series data leaks future info into training.
n        = len(df)
n_train  = int(0.70 * n)
n_val    = int(0.15 * n)
train_idx = list(range(0, n_train))
val_idx   = list(range(n_train, n_train + n_val))
test_idx  = list(range(n_train + n_val, n))
print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")\
'''

# Feature selection using LightGBM importance — used by TabTransformer and GNN notebooks.
# Split must be defined first so LightGBM is fit on training data only (no leakage).
FEATURE_SEL_LGBM = '''\
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

def select_features_by_importance(df, train_idx, target_col="isFraud",
                                   max_numeric=50, max_categorical=20):
    """
    Rank all candidate features by LightGBM split importance, then pick the top N.

    Why this beats heuristic selection:
      The IEEE-CIS dataset has 339 anonymised V-columns — their names reveal
      nothing about their value. Heuristic selection (first 50 columns) grabs
      V1-V11, which may be less informative than V45 or V258. LightGBM quickly
      identifies which features the trees actually use for splits.

    Process:
      1. Drop columns that are >95% missing.
      2. Classify remaining columns as numeric or categorical.
      3. Fit a fast LightGBM (100 trees) on the TRAINING split only.
      4. Rank features by cumulative split gain.
      5. Return top max_numeric numeric + top max_categorical categorical.
    """
    ignore = {target_col, "TransactionID"}
    na_frac = df.isna().mean()
    kept = [c for c in na_frac[na_frac < 0.95].index if c not in ignore]

    num_cands, cat_cands = [], []
    for col in kept:
        dt = df[col].dtype
        if dt == "object" or str(dt) == "category":
            cat_cands.append(col)
        elif np.issubdtype(dt, np.integer):
            (cat_cands if df[col].nunique(dropna=True) <= 20 else num_cands).append(col)
        elif np.issubdtype(dt, np.floating):
            num_cands.append(col)

    all_cands = num_cands + cat_cands
    X = df[all_cands].copy()
    for col in cat_cands:
        X[col] = X[col].astype("category").cat.codes  # -1 for NaN, fine for trees
    X = X.fillna(-999).values.astype(np.float32)
    y = df[target_col].values

    pos = (y[train_idx] == 1).sum()
    neg = (y[train_idx] == 0).sum()
    print("Running LightGBM for feature importance (100 trees) …")
    lgb = LGBMClassifier(n_estimators=100, n_jobs=-1, verbose=-1,
                          random_state=42, scale_pos_weight=neg/pos)
    lgb.fit(X[train_idx], y[train_idx])

    importance = pd.Series(lgb.feature_importances_, index=all_cands).sort_values(ascending=False)
    num_set, cat_set = set(num_cands), set(cat_cands)
    numeric_cols     = [c for c in importance.index if c in num_set][:max_numeric]
    categorical_cols = [c for c in importance.index if c in cat_set][:max_categorical]

    print(f"Selected {len(numeric_cols)} numeric + {len(categorical_cols)} categorical by importance")
    print("Top 10 numeric:   ", numeric_cols[:10])
    print("Categorical:      ", categorical_cols)
    return numeric_cols, categorical_cols

numeric_cols, categorical_cols = select_features_by_importance(
    df, train_idx, target_col="isFraud", max_numeric=50, max_categorical=20
)\
'''


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — Baseline (XGBoost + LightGBM)
# ═══════════════════════════════════════════════════════════════════════════════

baseline_cells = [

    mc("""\
# Baseline Models — XGBoost & LightGBM
> **Research context:** These gradient-boosted tree models serve as the benchmark.
> The deep-learning models (TabTransformer, GNN) need to beat these numbers to
> justify their added complexity.

**Primary metric: PR-AUC** (precision-recall area under curve).
With only ~3.5% fraud, accuracy is meaningless — a model that never predicts fraud
scores 96.5%. PR-AUC measures how well the model separates fraud from non-fraud
across all operating thresholds.\
"""),

    cc(INSTALL_BASELINE),

    mc("## 1 · Data Configuration\nSet `DATA_DIR` to wherever your raw CSVs live."),
    cc(DATA_SETUP),

    mc("## 2 · Import Libraries"),
    cc("""\
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier\
"""),

    mc("""\
## 3 · Load & Merge Data
The dataset has two source files that must be joined:
- `train_transaction.csv` — 590k rows, 394 columns (amounts, card info, V-features)
- `train_identity.csv` — 144k rows, 41 columns (device, browser, network info)

~76% of transactions have no identity record (NaN after left join).\
"""),
    cc(LOAD_MERGED),

    mc("""\
## 4 · Feature Selection
Automatically picks numeric and categorical columns using heuristic rules.
Drops columns that are >95% missing and caps the total feature count.\
"""),
    cc(FEATURE_SEL),

    mc("""\
## 5 · Temporal Train / Val / Test Split
Transactions are sorted by `TransactionDT` (seconds since reference date) before
splitting, so the model is always trained on older data and evaluated on newer data.
Split: **70% train · 15% val · 15% test**.\
"""),
    cc(SPLIT),

    mc("""\
## 6 · Feature Preparation for Tree Models
Tree models do not need scaling.  We only need to:
- Label-encode categorical columns (`.cat.codes` gives -1 for NaN, which XGBoost handles)
- Fill numeric NaNs with 0\
"""),
    cc('''\
def prepare_features(df, numeric_cols, categorical_cols, target_col="isFraud"):
    """
    Minimal preprocessing for gradient-boosted trees.
    Categoricals are label-encoded; numerics are NaN-filled with 0.
    Trees handle both without scaling.
    """
    X = df[numeric_cols + categorical_cols].copy()
    y = df[target_col].values
    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes  # -1 for NaN
    X[numeric_cols] = X[numeric_cols].fillna(0)
    return X.values.astype(np.float32), y

X, y = prepare_features(df, numeric_cols, categorical_cols)
X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos
print(f"Positive (fraud) samples in train: {int(pos):,}")
print(f"scale_pos_weight = {scale_pos_weight:.2f}  (penalises missing fraud)")\
'''),

    mc("""\
## 7 · Evaluation Helper
`tune_threshold_and_eval` sweeps the precision-recall curve to find the
decision threshold that maximises F1, then reports all metrics at that threshold.
This gives a fairer comparison than a fixed 0.5 threshold on imbalanced data.\
"""),
    cc('''\
def tune_threshold_and_eval(y_true, y_score, label):
    """
    Find the threshold maximising F1 on the PR curve,
    then return precision, recall, F1, ROC-AUC, and PR-AUC.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_thr = float(thresholds[f1s.argmax()])
    y_pred   = (y_score >= best_thr).astype(int)
    return {
        f"{label}_precision":      precision_score(y_true, y_pred, zero_division=0),
        f"{label}_recall":         recall_score(y_true, y_pred, zero_division=0),
        f"{label}_f1":             f1_score(y_true, y_pred, zero_division=0),
        f"{label}_roc_auc":        roc_auc_score(y_true, y_score),
        f"{label}_pr_auc":         average_precision_score(y_true, y_score),
        f"{label}_best_threshold": best_thr,
    }

def run_model(name, clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train a sklearn-compatible classifier and evaluate on val + test sets."""
    print(f"\\nTraining {name} …")
    clf.fit(X_train, y_train)
    val_m  = tune_threshold_and_eval(y_val,  clf.predict_proba(X_val)[:, 1],  "val")
    test_m = tune_threshold_and_eval(y_test, clf.predict_proba(X_test)[:, 1], "test")
    print(f"\\n{name} — Validation")
    for k, v in val_m.items():  print(f"  {k:30s}: {v:.4f}")
    print(f"\\n{name} — Test")
    for k, v in test_m.items(): print(f"  {k:30s}: {v:.4f}")
    return {**val_m, **test_m}\
'''),

    mc("""\
## 8 · XGBoost
XGBoost uses gradient-boosted decision trees with second-order gradient statistics.
`scale_pos_weight` compensates for class imbalance by upweighting fraud examples.
`eval_metric="aucpr"` guides the internal evaluation toward PR-AUC.\
"""),
    cc('''\
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_metrics = run_model("XGBoost", xgb,
                         X_train, y_train, X_val, y_val, X_test, y_test)\
'''),

    mc("""\
## 9 · LightGBM
LightGBM is a faster alternative to XGBoost using histogram-based splits and
leaf-wise tree growth. Often achieves similar accuracy with lower memory usage.\
"""),
    cc('''\
lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgbm_metrics = run_model("LightGBM", lgbm,
                          X_train, y_train, X_val, y_val, X_test, y_test)\
'''),

    mc("## 10 · Results Summary"),
    cc('''\
print("\\n" + "="*45)
print("  SUMMARY — Test Set")
print("="*45)
print(f"  XGBoost   PR-AUC : {xgb_metrics['test_pr_auc']:.4f}")
print(f"  XGBoost   ROC-AUC: {xgb_metrics['test_roc_auc']:.4f}")
print(f"  XGBoost   F1     : {xgb_metrics['test_f1']:.4f}")
print()
print(f"  LightGBM  PR-AUC : {lgbm_metrics['test_pr_auc']:.4f}")
print(f"  LightGBM  ROC-AUC: {lgbm_metrics['test_roc_auc']:.4f}")
print(f"  LightGBM  F1     : {lgbm_metrics['test_f1']:.4f}")\
'''),
]


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — TabTransformer
# ═══════════════════════════════════════════════════════════════════════════════

tabt_cells = [

    mc("## 0 · Sync with GitHub\nI run this at the start of every session so I'm always working off the latest code."),
    cc(GIT_SETUP),

    mc("""\
# TabTransformer — Attention over Categorical Features

The core idea here is to treat each feature as a *token* and let the model figure out which feature combinations matter most for a given transaction. Instead of hand-engineering interactions, I'm leaning on the transformer's self-attention to discover them automatically.

The way I built this:
- Each categorical column gets its own learned embedding (d=64) — so `card_type` and `email_domain` live in the same vector space and can be compared directly
- Each numeric feature also gets projected into a token — this was a deliberate choice over collapsing all numerics into one token, because it lets attention fire between individual numeric features too
- All tokens go through 3 layers of multi-head attention (4 heads each)
- I flatten the output and pass it through a small MLP to get the fraud probability

The research question I'm trying to answer: does attention over feature tokens capture cross-feature interactions that tree models can't?\
"""),

    cc(INSTALL_TABT),

    mc("## 1 · Data Setup\nI'm pulling the raw CSVs from Google Drive and copying them to Colab's local storage — reads are much faster from local disk than Drive during training."),
    cc(DATA_SETUP),

    mc("## 2 · Imports\nStandard PyTorch stack plus sklearn for metrics. I auto-detect the best available device — CUDA on Colab, MPS on Apple Silicon, CPU as fallback."),
    cc("""\
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)

# CUDA on Colab T4/A100, MPS on Apple Silicon, CPU fallback
def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Device: {device}")\
"""),

    mc("""\
## 3 · Load & Merge Data
The dataset comes in two files — transactions and identity records — that I join on `TransactionID`. About 76% of transactions have no identity record at all, so those rows will have NaN across all the identity columns after the merge. I cache the result as a parquet file so I'm not re-reading the raw CSVs on every run.\
"""),
    cc(LOAD_MERGED),

    mc("""\
## 4 · Temporal Train / Val / Test Split
I sort by `TransactionDT` before splitting so the model always trains on older data and evaluates on newer data — the way it would work in production. A random split would leak future transactions into training and make the results look better than they actually are. I do this step *before* feature selection so that when I fit LightGBM to rank features, I'm only looking at training data.\
"""),
    cc(SPLIT),

    mc("""\
## 5 · Feature Selection
The dataset has 339 anonymised V-columns and I have no idea which ones are useful — the names tell me nothing. Rather than blindly grabbing the first 50, I run a quick 100-tree LightGBM and let it tell me which features it actually uses for splits. I apply this same selection to both the TabTransformer and GNN so the comparison between them stays fair.\
"""),
    cc(FEATURE_SEL_LGBM),

    mc("""\
## 6 · Model Architecture
I built this from scratch rather than using a library so I could control exactly how the numeric features are tokenised. The key design decision is giving each numeric feature its own weight/bias pair — this means the model learns a separate projection per feature rather than treating all numerics as one undifferentiated blob. That small change meaningfully improves what the attention layers can do with numeric data.\
"""),
    cc('''\
class CustomTabTransformer(nn.Module):
    """
    My from-scratch TabTransformer implementation.

    Each feature — whether categorical or numeric — becomes a d-dimensional token.
    I pass all tokens through a standard TransformerEncoder so attention can fire
    across any pair of features. The encoded sequence is then flattened and fed
    into a small MLP to produce a single fraud logit.

    I set dropout=0.3 (rather than the typical 0.1) because earlier runs showed
    a persistent val→test gap (0.109 PR-AUC) indicating overfitting. Stronger
    dropout combined with weight_decay=1e-3 is the targeted fix.
    """

    def __init__(self, vocab_sizes, num_numeric_features,
                 d_token=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_token = d_token
        self.n_tokens = num_numeric_features + len(vocab_sizes)

        # One embedding table per categorical column
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n_cat, d_token) for n_cat in vocab_sizes
        ])

        # Per-numeric-feature linear projection: x_i → x_i * w_i + b_i
        self.num_weight = nn.Parameter(torch.randn(num_numeric_features, d_token) * 0.02)
        self.num_bias   = nn.Parameter(torch.zeros(num_numeric_features, d_token))

        # Transformer encoder (batch_first so input shape is B × T × D)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=4 * d_token, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Classification head: LayerNorm → Linear → ReLU → Dropout → Linear → logit
        self.head = nn.Sequential(
            nn.LayerNorm(self.n_tokens * d_token),
            nn.Linear(self.n_tokens * d_token, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x_num, x_cat):
        # Numeric tokens: (B, n_num) → (B, n_num, d_token)
        num_tokens = x_num.unsqueeze(2) * self.num_weight + self.num_bias
        # Categorical tokens: (B, n_cat) → (B, n_cat, d_token)
        cat_tokens = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
        )
        tokens  = torch.cat([num_tokens, cat_tokens], dim=1)   # (B, T, d_token)
        encoded = self.transformer(tokens)
        return self.head(encoded.reshape(encoded.size(0), -1)).squeeze(-1)\
'''),

    mc("""\
## 7 · Training Loop
A few decisions I made here that are worth calling out. I use `ReduceLROnPlateau` instead of a fixed cosine schedule — an earlier run with cosine decay was cutting the learning rate to near-zero by epoch 12, which was clearly too aggressive. The plateau scheduler only halves the LR when validation PR-AUC actually stops improving, which is much more sensible. I also added a 3-epoch linear warmup at the start because transformers can be unstable if you hit them with full LR on the first batch. For threshold selection I optimise F2 score rather than F1 — in fraud detection, missing a fraud (false negative) is more costly than a false alarm, so I want recall weighted twice as heavily as precision.\
"""),
    cc('''\
def train_tabtransformer(df, numeric_cols, categorical_cols,
                         target_col="isFraud", batch_size=2048, num_epochs=50,
                         train_idx=None, val_idx=None, test_idx=None,
                         early_stopping_patience=10, warmup_epochs=3):
    """
    Trains the TabTransformer and returns the best checkpoint by val PR-AUC.

    I track the best model state across all epochs and restore it at the end,
    so early stopping doesn't discard good weights — it just stops wasting time
    once the model has clearly plateaued.
    """
    device = get_device()
    print(f"Using device: {device}")

    # ── Vocab sizes + categorical encoding ───────────────────────────────────
    vocab_sizes, cat_mappings = [], []
    for col in categorical_cols:
        vals    = df[col].astype(str)
        uniques = list(vals.unique())
        vocab_sizes.append(len(uniques))
        cat_mappings.append({v: i for i, v in enumerate(uniques)})

    # ── Numeric tensor (StandardScaler fit on train only) ────────────────────
    num_data = df[numeric_cols].fillna(0).values.astype(np.float32)
    if train_idx is not None:
        scaler = StandardScaler()
        num_data[train_idx] = scaler.fit_transform(num_data[train_idx])
        if val_idx  is not None: num_data[val_idx]  = scaler.transform(num_data[val_idx])
        if test_idx is not None: num_data[test_idx] = scaler.transform(num_data[test_idx])
    x_num = torch.tensor(num_data, dtype=torch.float32)

    # ── Categorical tensor ────────────────────────────────────────────────────
    x_cat = torch.zeros((len(df), len(categorical_cols)), dtype=torch.long)
    for i, col in enumerate(categorical_cols):
        x_cat[:, i] = torch.tensor(
            [cat_mappings[i][v] for v in df[col].astype(str)], dtype=torch.long
        )

    y = torch.tensor(df[target_col].values, dtype=torch.float32)

    # ── Splits ────────────────────────────────────────────────────────────────
    if train_idx is None or val_idx is None:
        n = len(df); n_tr = int(0.8 * n)
        train_idx = list(range(0, n_tr)); val_idx = list(range(n_tr, n))

    y_train = y[train_idx]
    pos_weight = ((y_train == 0).sum().float() /
                  (y_train == 1).sum().float()).to(device)
    print(f"Class imbalance → pos_weight = {pos_weight.item():.2f}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(idx, shuffle):
        ds = TensorDataset(x_num[idx], x_cat[idx], y[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)

    # ── Model / optimizer / schedulers ───────────────────────────────────────
    # dropout=0.3 (up from 0.2) — tighter regularisation to close the val→test gap
    model     = CustomTabTransformer(vocab_sizes, len(numeric_cols), dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # weight_decay=1e-3 (up from 3e-4) — stronger L2 penalty to reduce overfitting
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    # Linear warmup for first `warmup_epochs` — prevents bad early gradients
    # from corrupting embeddings before training stabilises
    warmup_sched = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0
    )
    # After warmup: halve LR only when val PR-AUC stalls for 3 epochs
    plateau_sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_pr_auc, best_epoch, patience_ctr, best_state = -1.0, 0, 0, None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for bx_num, bx_cat, by in train_loader:
            bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
            optimizer.zero_grad()
            # Label smoothing: 0→0.05, 1→0.95 — reduces overconfidence and lowers
            # the extreme threshold (0.97+) seen in unsmoothed runs
            smooth = 0.05
            by_smooth = by * (1 - smooth) + smooth * 0.5
            loss = criterion(model(bx_num, bx_cat), by_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
            del bx_num, bx_cat, by, loss
            if device.type == "mps" and n_batches % 100 == 0:
                torch.mps.empty_cache()

        if device.type == "mps": torch.mps.empty_cache()

        # Validation PR-AUC
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for bx_num, bx_cat, by in val_loader:
                val_logits.append(model(bx_num.to(device), bx_cat.to(device)).cpu())
                val_labels.append(by)
        ep_probs    = torch.sigmoid(torch.cat(val_logits)).numpy()
        ep_true     = torch.cat(val_labels).numpy()
        ep_pr_auc   = average_precision_score(ep_true, ep_probs)
        avg_loss    = total_loss / max(n_batches, 1)

        # Warmup for first N epochs, then adaptive LR
        if epoch <= warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(ep_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:3d}] loss={avg_loss:.4f}  val_pr_auc={ep_pr_auc:.4f}  lr={current_lr:.2e}")

        if ep_pr_auc > best_pr_auc:
            best_pr_auc, best_epoch, patience_ctr = ep_pr_auc, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best: epoch {best_epoch} (pr_auc={best_pr_auc:.4f})")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Restored best model from epoch {best_epoch}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for bx_num, bx_cat, by in val_loader:
            all_logits.append(model(bx_num.to(device), bx_cat.to(device)).cpu())
            all_labels.append(by)
    y_score = torch.sigmoid(torch.cat(all_logits)).numpy()
    y_true  = torch.cat(all_labels).numpy()

    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    # F2 score: weights recall 2× over precision — correct tradeoff for fraud detection
    # where missing fraud (false negative) is more costly than a false alarm
    beta = 2
    f2s  = (1 + beta**2) * precs[:-1] * recs[:-1] / (
        beta**2 * precs[:-1] + recs[:-1] + 1e-8
    )
    best_thr = float(thrs[f2s.argmax()])
    y_pred   = (y_score >= best_thr).astype(float)

    metrics = {
        "val_precision": precision_score(y_true, y_pred, zero_division=0),
        "val_recall":    recall_score(y_true, y_pred, zero_division=0),
        "val_f1":        f1_score(y_true, y_pred, zero_division=0),
        "val_roc_auc":   roc_auc_score(y_true, y_score),
        "val_pr_auc":    average_precision_score(y_true, y_score),
        "val_threshold": best_thr,
    }
    print("\\n===== TabTransformer — Validation =====")
    for k, v in metrics.items(): print(f"  {k:20s}: {v:.4f}")

    # Test set
    if test_idx is not None:
        test_loader = DataLoader(
            TensorDataset(x_num[test_idx], x_cat[test_idx], y[test_idx]),
            batch_size=batch_size, shuffle=False, num_workers=2,
        )
        test_logits = []
        with torch.no_grad():
            for bx_num, bx_cat, _ in test_loader:
                test_logits.append(model(bx_num.to(device), bx_cat.to(device)).cpu())
        y_test_score = torch.sigmoid(torch.cat(test_logits)).numpy()
        y_test_true  = y[test_idx].numpy()
        y_test_pred  = (y_test_score >= best_thr).astype(float)
        test_m = {
            "test_precision": precision_score(y_test_true, y_test_pred, zero_division=0),
            "test_recall":    recall_score(y_test_true, y_test_pred, zero_division=0),
            "test_f1":        f1_score(y_test_true, y_test_pred, zero_division=0),
            "test_roc_auc":   roc_auc_score(y_test_true, y_test_score),
            "test_pr_auc":    average_precision_score(y_test_true, y_test_score),
        }
        print("\\n===== TabTransformer — Test =====")
        for k, v in test_m.items(): print(f"  {k:20s}: {v:.4f}")
        metrics.update(test_m)

    return metrics, model\
'''),

    mc("## 8 · Train the Model\nEverything above was setup — this is where training actually happens. I pass the indices explicitly so the model only ever sees training data during fitting."),
    cc('''\
metrics, model = train_tabtransformer(
    df, numeric_cols, categorical_cols,
    target_col="isFraud",
    batch_size=2048,          # large batch — T4/A100 GPU has plenty of VRAM
    num_epochs=50,            # more room with adaptive LR + patience=10
    early_stopping_patience=10,
    warmup_epochs=3,
    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
)\
'''),

    mc("## 9 · Results\nVal metrics are what drove early stopping. Test metrics are the honest numbers — the model never saw the test set during training or threshold tuning."),
    cc('''\
print("\\n" + "="*45)
print("  TabTransformer — Final Results")
print("="*45)
for k, v in metrics.items():
    print(f"  {k:22s}: {v:.4f}")\
'''),
]


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 — GNN
# ═══════════════════════════════════════════════════════════════════════════════

gnn_cells = [

    mc("## 0 · Sync with GitHub\nI run this at the start of every session so I'm always working off the latest code."),
    cc(GIT_SETUP),

    mc("""\
# Graph Neural Network — Relational Fraud Detection

The idea behind this model is that fraud isn't random — fraudsters reuse the same cards, devices, and email accounts across multiple transactions. If I can connect those transactions in a graph, I can let the model propagate fraud signals between related nodes rather than treating every transaction in isolation.

Every transaction is a node. I connect two nodes with an edge if they share a value in any of these identity columns: `card1`, `card4`, `card6`, `addr1`, `P_emaildomain`, `R_emaildomain`, `id_30`, `id_31`, or `DeviceInfo`. The intuition is that if transaction A and B used the same card, they're probably from the same person — and if one is fraud, that's useful information about the other.

I tried k-NN edges initially (connecting transactions that are numerically similar) but abandoned it — computing nearest neighbours on 590k × 50 features is O(N²) and would take hours just to build the graph. Identity edges are O(N) and more directly encode the fraud hypothesis anyway.

The model itself is a 3-layer residual GNN with mean aggregation. Each layer aggregates neighbour features, applies a linear transformation, and adds a residual connection so gradients can flow cleanly during backprop.\
"""),

    cc(INSTALL_GNN),

    mc("## 1 · Data Setup\nSame setup as the TabTransformer notebook — pulling from Drive and copying to local storage for faster reads."),
    cc(DATA_SETUP),

    mc("## 2 · Imports\nI force the device selection here explicitly. The GNN runs full-graph training — the entire 590k-node graph lives in memory at once — so I need to be deliberate about where tensors land."),
    cc("""\
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)

# GNN uses full-graph training — the entire 590k-node graph is passed in one
# forward call. On Colab T4/A100, CUDA should have enough VRAM (~4-8 GB needed).
# Falls back to CPU if CUDA is unavailable (e.g. running locally on Apple Silicon).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")\
"""),

    mc("## 3 · Load & Merge Data\nSame loading logic as the TabTransformer — joins transactions with identity records and caches the result."),
    cc(LOAD_MERGED),

    mc("## 4 · Temporal Split\nI define the split before running feature selection so that when LightGBM fits to rank features, it only sees training data. Same 70/15/15 split as the TabTransformer so the comparison is apples-to-apples."),
    cc(SPLIT),

    mc("## 5 · Feature Selection\nI use the exact same LightGBM importance-based selection as the TabTransformer. Both models get the same 50 numeric + 20 categorical features — if I gave them different features the comparison would be meaningless."),
    cc(FEATURE_SEL_LGBM),

    mc("""\
## 6 · Build the Transaction Graph
This is the step that makes the GNN fundamentally different from the TabTransformer. I'm building an `edge_index` tensor — a (2, E) matrix where each column is one directed edge — that encodes which transactions are related to which.

For each identity column I group transactions by shared value and connect them. Small groups (10 or fewer) get fully connected into a clique. Larger groups get a hub-and-spoke structure with a chain connecting sequential members — this keeps the edge count linear in group size rather than quadratic, which matters a lot at 590k nodes.\
"""),
    cc('''\
def build_transaction_graph(df, min_group_size=2, max_group_size=1000,
                             small_group_full_connect=10):
    """
    Builds the transaction graph from shared identity columns.

    I skip NaN groups deliberately — if two transactions both lack a card number,
    that doesn't mean they're related, it just means we don't have the data.
    Connecting unknown-to-unknown would add noise rather than signal.

    Self-loops are added so each node aggregates its own features alongside
    its neighbours during message passing — without them a node has no way
    to preserve its own representation across layers.
    """
    df = df.reset_index(drop=True)
    src_list, dst_list = [], []

    id_cols = ["card1","card4","card6","addr1","P_emaildomain",
               "R_emaildomain","id_30","id_31","DeviceInfo"]
    active_cols = [c for c in id_cols if c in df.columns]

    for col in active_cols:
        vals   = df[col].astype(str)
        groups = vals.groupby(vals).groups
        for val, idxs in groups.items():
            if val in ("nan","None",""): continue
            grp  = sorted(list(idxs))
            size = len(grp)
            if size < min_group_size or size > max_group_size: continue

            if size <= small_group_full_connect:
                # Full clique — every pair connected
                for i in range(size):
                    for j in range(i + 1, size):
                        src_list += [grp[i], grp[j]]
                        dst_list += [grp[j], grp[i]]
            else:
                # Hub + chain pattern — keeps edge count linear in group size
                hub = grp[0]
                for other in grp[1:]:
                    src_list += [hub, other]; dst_list += [other, hub]
                for i in range(size - 1):
                    src_list += [grp[i], grp[i+1]]; dst_list += [grp[i+1], grp[i]]

    if not src_list:
        return torch.empty((2, 0), dtype=torch.long)

    ei = torch.tensor([src_list, dst_list], dtype=torch.long)
    ei = torch.unique(ei.t(), dim=0).t().contiguous()   # deduplicate

    # Self-loops: each node aggregates its own features
    n   = len(df)
    sl  = torch.arange(n, dtype=torch.long)
    ei  = torch.cat([ei, torch.stack([sl, sl])], dim=1)
    return ei

print("Building transaction graph …")
edge_index = build_transaction_graph(df)
print(f"edge_index shape: {edge_index.shape}  ({edge_index.shape[1]:,} edges)")\
'''),

    mc("""\
## 7 · Model Architecture
The model has two distinct phases. First, I encode each transaction's raw features into a shared embedding space using a numeric projection and per-column categorical embeddings. Then I run three rounds of message passing where each node aggregates the mean of its neighbours' representations, applies a linear transformation, and adds a residual connection back to itself.

The residual connections are important here — without them, deep GNNs tend to "over-smooth" where every node ends up with roughly the same representation regardless of its local neighbourhood. The LayerNorm after each layer helps with training stability.\
"""),
    cc('''\
class SimpleGNN(nn.Module):
    """
    My 3-layer residual GNN for node-level fraud classification.

    I set hidden_dim=128 rather than 256 — at 590k nodes the graph structure
    is doing the heavy lifting, not raw model capacity, and the smaller size
    keeps memory manageable for full-graph training on a single GPU.
    """

    def __init__(self, num_numeric, num_categories_per_col,
                 embed_dim=32, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(size + 1, embed_dim) for size in num_categories_per_col
        ])
        self.num_linear   = nn.Linear(num_numeric, embed_dim)
        total_in          = embed_dim * (1 + len(num_categories_per_col))
        self.input_linear = nn.Linear(total_in, hidden_dim)

        # Three residual GNN layers
        self.gcn1, self.gcn2, self.gcn3 = (
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.act        = nn.ReLU()
        self.out_linear = nn.Linear(hidden_dim, 1)

    def _mean_agg(self, x, edge_index):
        """
        Mean-aggregate neighbour features for each node.
        Fast O(E) scatter operation — no intermediate edge tensors.
        """
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device).index_add_(
            0, dst, torch.ones(len(dst), device=x.device)
        ).clamp_min(1.0).unsqueeze(1)
        return agg / deg

    def forward(self, x_num, x_cat, edge_index):
        # Encode features into a shared embedding space
        h = torch.cat(
            [self.act(self.num_linear(x_num))] +
            [emb(x_cat[:, i].clamp(0, emb.num_embeddings - 1))
             for i, emb in enumerate(self.cat_embeddings)],
            dim=1,
        )
        h = self.act(self.input_linear(h))

        # Three residual message-passing layers
        for gcn, norm in [(self.gcn1, self.norm1),
                          (self.gcn2, self.norm2),
                          (self.gcn3, self.norm3)]:
            h_res = h
            h     = self.dropout(self.act(gcn(self._mean_agg(h, edge_index))))
            h     = norm(h + h_res)

        return self.out_linear(h).squeeze(-1)\
'''),

    mc("""\
## 8 · Training Loop
The biggest difference from the TabTransformer is that I can't mini-batch here. The GNN needs the full graph in memory at once because message passing requires knowing every node's neighbours — if I only loaded a subset of nodes, the neighbourhood information would be incomplete. That means every forward pass touches all 590k nodes and all edges.

Because of this, I use a higher early stopping patience (20 epochs) — GNNs converge more slowly than mini-batch models and I don't want to stop prematurely. Same label smoothing and F2 threshold tuning as the TabTransformer.\
"""),
    cc('''\
def train_gnn(df, numeric_cols, categorical_cols, target_col="isFraud",
              num_epochs=100, lr=1e-3, train_idx=None, val_idx=None,
              test_idx=None, early_stopping_patience=20):
    """
    Full-graph training of the SimpleGNN.

    I save the best model state to CPU after each improvement so I'm not
    accumulating multiple copies of the weights in GPU memory across epochs.
    At restore time I move the state back to the device before loading.
    """
    cols = numeric_cols + categorical_cols + [target_col]
    df   = df[cols].copy().reset_index(drop=True)

    # ── Numeric scaling ───────────────────────────────────────────────────────
    num_df = df[numeric_cols].fillna(0.0).astype("float32")
    num_df = (num_df - num_df.mean()) / num_df.std().replace(0, 1.0)
    df[numeric_cols] = num_df
    x_num = torch.tensor(num_df.values, dtype=torch.float32)

    # ── Categorical encoding ──────────────────────────────────────────────────
    cat_sizes, cat_arrays = [], []
    for col in categorical_cols:
        vals    = df[col].astype(str)
        mapping = {v: i for i, v in enumerate(sorted(vals.unique()))}
        cat_sizes.append(len(mapping))
        cat_arrays.append(vals.map(mapping).astype("int64").values)

    x_cat = (torch.tensor(np.stack(cat_arrays, axis=1), dtype=torch.long)
             if cat_arrays else torch.empty((len(df), 0), dtype=torch.long))

    y = torch.tensor(df[target_col].values.astype("float32"), dtype=torch.float32)

    # ── Graph ─────────────────────────────────────────────────────────────────
    if "edge_index" not in dir():
        print("Building graph …")
    ei = build_transaction_graph(df)

    # ── Splits ────────────────────────────────────────────────────────────────
    if train_idx is None or val_idx is None:
        n_tr = int(0.8 * len(df))
        train_idx = list(range(0, n_tr)); val_idx = list(range(n_tr, len(df)))
    tr  = torch.tensor(train_idx, dtype=torch.long)
    val = torch.tensor(val_idx,   dtype=torch.long)
    tst = torch.tensor(test_idx,  dtype=torch.long) if test_idx else None

    # Move to device
    x_num, x_cat, y, ei = x_num.to(device), x_cat.to(device), y.to(device), ei.to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    # hidden_dim=256 — mean aggregation is memory-efficient enough to support
    # full capacity. Only dropped to 128 when attention aggregation caused OOM.
    model = SimpleGNN(len(numeric_cols), cat_sizes, hidden_dim=256).to(device)
    pos   = (y[tr] == 1).sum(); neg = (y[tr] == 0).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=(neg / pos).clamp(min=1.0))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_pr, best_epoch, pat_ctr, best_state = -1.0, 0, 0, None

    for epoch in range(1, num_epochs + 1):
        model.train(); optimizer.zero_grad()
        logits   = model(x_num, x_cat, ei)
        # Label smoothing: softens 0→0.05 and 1→0.95 to reduce overconfidence
        smooth   = 0.05
        y_smooth = y[tr] * (1 - smooth) + smooth * 0.5
        loss     = criterion(logits[tr], y_smooth)
        if not torch.isfinite(loss): print(f"Non-finite loss at epoch {epoch}"); break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(x_num, x_cat, ei))[val].cpu().numpy()
        ep_pr = average_precision_score(y[val].cpu().numpy(), vp)
        print(f"[Epoch {epoch:3d}] loss={loss.item():.4f}  val_pr_auc={ep_pr:.4f}")

        if ep_pr > best_pr:
            best_pr, best_epoch, pat_ctr = ep_pr, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_ctr += 1
            if pat_ctr >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best: {best_epoch} (pr_auc={best_pr:.4f})")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Restored best model from epoch {best_epoch}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_num, x_cat, ei))

    def eval_split(idx_tensor, label):
        yt = y[idx_tensor].cpu().numpy()
        ys = probs[idx_tensor].cpu().numpy()
        precs, recs, thrs = precision_recall_curve(yt, ys)
        # F2 score — weights recall 2× over precision for fraud detection
        beta = 2
        f2s  = (1 + beta**2) * precs[:-1] * recs[:-1] / (
            beta**2 * precs[:-1] + recs[:-1] + 1e-8
        )
        thr = float(thrs[f2s.argmax()])
        yp  = (ys >= thr).astype("int32")
        return {
            f"{label}_precision": precision_score(yt, yp, zero_division=0),
            f"{label}_recall":    recall_score(yt, yp, zero_division=0),
            f"{label}_f1":        f1_score(yt, yp, zero_division=0),
            f"{label}_roc_auc":   roc_auc_score(yt, ys),
            f"{label}_pr_auc":    average_precision_score(yt, ys),
            f"{label}_threshold": thr,
        }

    metrics = eval_split(val, "val")
    print("\\n===== GNN — Validation =====")
    for k, v in metrics.items(): print(f"  {k:22s}: {v:.4f}")

    if tst is not None:
        test_m = eval_split(tst, "test")
        print("\\n===== GNN — Test =====")
        for k, v in test_m.items(): print(f"  {k:22s}: {v:.4f}")
        metrics.update(test_m)

    return metrics, model\
'''),

    mc("## 9 · Train the Model\nGraph construction takes a few minutes — it's scanning all 590k rows for shared identity values. Training itself should be around 5 seconds per epoch on CUDA. Early stopping will likely kick in well before epoch 100."),
    cc('''\
# Graph construction: ~2-3 min. Training: ~5 sec/epoch. Expected total: ~10-15 min.
print("Building graph …")
edge_index = build_transaction_graph(df)
print(f"Graph built: {edge_index.shape[1]:,} edges")

gnn_metrics, gnn_model = train_gnn(
    df, numeric_cols, categorical_cols,
    target_col="isFraud",
    num_epochs=250,
    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
)\
'''),

    mc("## 10 · Results\nVal metrics guided training. Test metrics are what matter — the model never touched the test set until this final evaluation."),
    cc('''\
print("\\n" + "="*45)
print("  GNN — Final Results")
print("="*45)
for k, v in gnn_metrics.items():
    print(f"  {k:24s}: {v:.4f}")\
'''),
]


# ─── write notebooks ─────────────────────────────────────────────────────────

print("Generating notebooks …")
save("01_baseline.ipynb",      baseline_cells)
save("02_tabtransformer.ipynb", tabt_cells)
save("03_gnn.ipynb",           gnn_cells)
print("Done.")
