# DSC 599 — Fraud Detection: TabTransformer vs GNN

This is my capstone project for DSC 599. The research question I'm investigating is whether modern deep learning architectures — specifically attention-based and graph-based models — capture fraud patterns in financial transaction data that traditional methods can't. I built both models entirely from scratch so I could understand every design decision, rather than treating them as black boxes.

The dataset is the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset: 590,540 transactions, 3.5% fraud rate, 434 features after merging the transaction and identity tables.

---

## The Two Models

### TabTransformer
I treat every feature as a token and let transformer attention learn which feature combinations are suspicious. Each categorical column gets an embedding, each numeric feature gets its own linear projection, and all 70 tokens flow through 3 transformer encoder layers before a classification head produces a fraud probability. The key insight is that attention can discover interactions like "card type X + email domain Y + device Z" without me telling it those combinations matter.

### GNN
I build a graph where each transaction is a node, and two nodes are connected if they share an identity attribute — same card, same email domain, same device, etc. The intuition is that fraudsters reuse payment instruments, so if card X was used fraudulently once, every other transaction on that card deserves elevated suspicion. A 3-layer residual GNN propagates these signals across connected nodes through mean aggregation.

---

## Results

Both models use the same 70/15/15 temporal split (sorted by `TransactionDT`) and the same LightGBM importance-ranked feature set (50 numeric + 20 categorical). Primary metric is **PR-AUC** — accuracy is meaningless at 3.5% fraud rate.

| Metric | TabTransformer | GNN |
|---|---|---|
| **test PR-AUC** | **0.4140** | 0.3437 |
| test ROC-AUC | 0.8314 | **0.8317** |
| test Recall | 0.5219 | **0.5644** |
| test Precision | **0.3136** | 0.2012 |
| test F1 | **0.3918** | 0.2967 |

TabTransformer wins on most metrics. The GNN catches slightly more fraud (higher recall) but with more false positives. Both perform roughly 10–12× above random (random PR-AUC ≈ 0.035). For a full breakdown see [`ARCHITECTURE_REPORT.md`](ARCHITECTURE_REPORT.md).

---

## Project Structure

```
├── src/
│   ├── data_loading.py                    # Merge + cache the two raw CSVs
│   ├── feature_selection.py               # LightGBM importance-based feature ranking
│   ├── utils_metrics.py                   # Shared device detection utility
│   │
│   ├── tabtransformer_custom/
│   │   ├── model_custom.py                # CustomTabTransformer architecture
│   │   └── train_custom.py                # Training loop, early stopping, F2 threshold
│   │
│   └── gnn_custom/
│       ├── graph_utils.py                 # Identity-edge graph construction
│       ├── gnn_model.py                   # 3-layer residual GNN
│       └── train_gnn.py                   # Full-graph training loop
│
├── run_tabTransformer.py                  # Entry point — TabTransformer
├── run_gnn.py                             # Entry point — GNN
├── run_baseline.py                        # XGBoost + LightGBM baselines
├── ARCHITECTURE_REPORT.md                 # Full architecture + performance writeup
└── requirements.txt
```

---

## Running Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw data files at:
```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

Then run either model:
```bash
python run_tabTransformer.py
python run_gnn.py
```

Logs are written to `results/logs/`. The first run merges and caches the data to `data/interim/merged_train.parquet` so subsequent runs load instantly.

---

## Key Design Decisions Worth Noting

**Why I dropped k-NN edges from the GNN:** Computing nearest neighbours on 590k × 50 features is O(N²) — it would take hours just to build the graph. Identity edges are O(N) and more directly encode the fraud hypothesis anyway.

**Why PR-AUC over accuracy:** At 3.5% fraud rate, a model that predicts "not fraud" on everything scores 96.5% accuracy. PR-AUC measures the precision-recall tradeoff across all thresholds and is robust to class imbalance.

**Why F2 for threshold tuning:** In fraud detection, missing a fraud (false negative) is more costly than a false alarm. F2 weights recall twice as heavily as precision when finding the optimal operating threshold.

**Why full-graph training for the GNN:** Message passing requires knowing every node's neighbours simultaneously. Mini-batching would require graph partitioning (e.g. PyTorch Geometric's NeighborSampler), which is a significant rewrite. Full-graph training works fine on a GPU with enough VRAM.

---

## The `colab` Branch

There's a `colab` branch with self-contained Jupyter notebooks for running everything in Google Colab. Each notebook handles its own data loading, feature selection, model definition, and training — no local setup required beyond uploading the raw CSVs to Google Drive.
