# DSC 599 — Fraud Detection: Colab Notebooks

This is the `colab` branch of my DSC 599 capstone project. Everything here is packaged into self-contained Jupyter notebooks designed to run in Google Colab — no local Python environment, no manual imports, no src/ directory to manage. Each notebook pulls the latest code from this branch at startup and runs top to bottom.

For the full architecture writeup and performance breakdown, see [`ARCHITECTURE_REPORT.md`](ARCHITECTURE_REPORT.md).

---

## What's in Here

```
notebooks/
├── 01_baseline.ipynb          # XGBoost + LightGBM baselines
├── 02_tabtransformer.ipynb    # Full TabTransformer pipeline
└── 03_gnn.ipynb               # Full GNN pipeline
```

Each notebook is completely self-contained — the model definitions, training loops, data loading, and feature selection are all inlined with documentation explaining each decision. I did this intentionally so the notebooks work in Colab without needing to upload the `src/` directory.

---

## How to Run

### 1. Get the data onto Google Drive
The IEEE-CIS competition is closed so the Kaggle API won't work. Upload both raw files to a folder in your Drive:
- `train_transaction.csv`
- `train_identity.csv`

I put mine in `My Drive/Data_Fraud/`. If you use a different folder name, update the `DRIVE_DATA_PATH` line in the data setup cell.

### 2. Open a notebook in Colab
File → Open notebook → GitHub tab → search `ConlerSimmons/DSC_599_Semester_Project` → switch branch to `colab` → pick a notebook.

### 3. Set your runtime
Runtime → Change runtime type → **A100 GPU** (with Colab Pro). The TabTransformer runs in ~15–20 minutes. The GNN takes ~10–15 minutes.

### 4. Run cell 0 first
The very first cell in each notebook syncs with this GitHub branch. If the repo is already cloned it pulls the latest; if not it clones fresh. This means whenever I push an update here, you just re-run cell 0 and you're current — no re-uploading.

### 5. Run the rest top to bottom
Runtime → Run all, or Shift+Enter through each cell. The data setup cell will ask you to authorise Google Drive access — click through and it handles everything else automatically.

---

## What Each Notebook Does

### `02_tabtransformer.ipynb`
1. Syncs with GitHub
2. Installs dependencies
3. Mounts Drive and copies data to Colab local storage
4. Loads and merges the two CSV files
5. Creates a 70/15/15 temporal train/val/test split
6. Runs LightGBM to rank features by importance and selects the top 50 numeric + 20 categorical
7. Defines the CustomTabTransformer model
8. Trains with AdamW + warmup + ReduceLROnPlateau, early stopping on val PR-AUC
9. Evaluates on val and test sets with F2 threshold tuning
10. Prints final metrics

### `03_gnn.ipynb`
1. Syncs with GitHub
2. Installs dependencies
3. Mounts Drive and copies data
4. Loads and merges the two CSV files
5. Creates the same 70/15/15 split
6. Runs the same LightGBM feature selection
7. Builds the transaction graph from shared identity attributes (card, email, device, etc.)
8. Defines the SimpleGNN model (3-layer residual, mean aggregation)
9. Trains full-graph with AdamW, early stopping on val PR-AUC
10. Evaluates with F2 threshold tuning
11. Prints final metrics

---

## Current Results

| Metric | TabTransformer | GNN |
|---|---|---|
| **test PR-AUC** | **0.4383** | 0.3695 |
| test ROC-AUC | 0.8373 | **0.8476** |
| test Recall | 0.5037 | **0.5388** |
| test Precision | **0.3764** | 0.2330 |
| test F1 | **0.4309** | 0.3253 |

Both models are roughly 10–12× better than random (random PR-AUC ≈ 0.035 at 3.5% fraud rate). TabTransformer wins on most metrics; GNN wins on recall and ROC-AUC.

---

## Differences from the `main` Branch

The `main` branch has the same models as `.py` scripts under `src/` — that's the local development version. This `colab` branch has:
- Everything inlined into notebooks (no src/ imports)
- LightGBM importance-based feature selection (vs heuristic first-N columns in main)
- Larger batch size (2048 vs 512) scaled for GPU
- ReduceLROnPlateau with linear warmup (vs CosineAnnealingLR in main)
- F2 threshold tuning (vs F1 in main)
- Label smoothing (not in main)
- hidden_dim=256 for GNN with CUDA support
