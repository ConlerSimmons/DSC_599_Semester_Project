# DSC 599 – Fraud Detection Using Deep Tabular Models  
### Custom TabTransformer + Graph Neural Network Pipelines

This project is the final capstone work for my DSC 599 course. I built and compared two modern deep learning architectures for fraud detection on the IEEE-CIS dataset: a **Custom TabTransformer** and a **Custom Graph Neural Network (GNN)**. Both models were created entirely from scratch and trained on highly imbalanced financial transaction data.

---

## Project Overview

The goal of this project was to understand how different modern architectures handle structured, high-dimensional, and extremely imbalanced tabular data. Fraud detection is a domain where **relationships** between features often matter more than the features themselves, and the signal is subtle and rare.

I implemented two pipelines:

1. **TabTransformer** – A model that uses **transformer attention** to capture cross-feature interactions among categorical variables while projecting numeric variables through an MLP.
2. **Graph Neural Network (GNN)** – A model that treats each transaction as a **node in a graph**, connecting nodes via similarity (k-NN) and shared identity/device attributes.

Each model is trained independently and produces separate evaluation metrics, which I compare in my final report and presentation.

---

## Directory Structure

```
DSC_599_Semester_Project
│
├── data/
│   └── raw/              # IEEE-CIS fraud dataset (train_transaction + train_identity)
│
├── src/
│   ├── data_loading.py   # Handles merging + preprocessing of original Kaggle files
│   ├── feature_selection.py
│   ├── utils_metrics.py
│   │
│   ├── tabtransformer_custom/
│   │   ├── model_custom.py     # Custom TabTransformer implementation
│   │   └── train_custom.py     # Training loop + metrics for the TabTransformer
│   │
│   └── gnn_custom/
│       ├── graph_utils.py      # Graph construction (kNN + identity/device edges)
│       ├── gnn_model.py        # Multi-layer GNN with residuals, dropout, layer norm
│       └── train_gnn.py        # Training pipeline + adaptive thresholding
│
├── run_tabTransformer.py       # Runner script for the custom TabTransformer
├── run_gnn.py                  # Runner script for the custom GNN
│
├── lightning_logs/             # TabTransformer training logs
├── saved_models/               # Serialized model checkpoints
│
└── requirements.txt
```

---

## System Architecture

### **1. Data Loading + Preprocessing**
All models use a unified loading function:

- Merges `train_transaction.csv` and `train_identity.csv`
- Handles missing values
- Standardizes numeric features
- Converts categorical strings into indices
- Maintains consistent row ordering so node IDs map cleanly in the GNN

---

# TabTransformer Architecture

My **TabTransformer** follows the design proposed by Google Research but implemented manually:

- **Categorical embeddings** (one per column)
- **Transformer encoder layers** to learn cross-feature interactions  
- **Numeric features** projected through an MLP
- Concatenation → classification head
- **Class imbalance handling** with `pos_weight`
- **Evaluation metrics**: precision, recall, F1, ROC-AUC, PR-AUC

This model tends to **favor recall** because attention layers let it “spread” anomaly signals across categorical interactions.

To run:

```
python run_tabTransformer.py
```

---

# 🕸️ Graph Neural Network (GNN) Architecture

The GNN pipeline has three major parts:

---

## **A. Graph Construction (`graph_utils.py`)**

Each transaction becomes a node.  
Edges come from:

1. **k-Nearest Neighbor edges (numeric similarity)**  
2. **Identity/device clusters**  
   - `card1`
   - `addr1`
   - `P_emaildomain`
   - `id_30` (OS)
   - `id_31` (browser)
   - `DeviceInfo`

Small groups become full cliques, mid-sized groups form hub-and-chain patterns, and very large clusters are ignored to avoid blowing up the graph.

Self-loops are added for stability.

---

## **B. GNN Model (`gnn_model.py`)**

The final architecture uses:

- **Numeric + categorical embeddings**
- **Two GCN-style message passing layers**
- **Residual connections**
- **Layer normalization**
- **Dropout**
- **A final linear classifier**

This architecture was chosen because deeper GCNs quickly oversmooth on this dataset and because the identity-based graph structure benefits from residual learning.

---

## **C. Training Pipeline (`train_gnn.py`)**

Features of the training loop:

- Full-batch gradient descent (graph-sized)
- `BCEWithLogitsLoss` with **dynamic pos_weight**
- 50 epochs by default
- **Best threshold search** using precision-recall curve
- Output:  
  - precision  
  - recall  
  - F1  
  - ROC-AUC  
  - PR-AUC  
  - learned optimal threshold

To run:

```
python run_gnn.py
```

---

# Model Behavior Summary

From the latest evaluations:

### **GNN**
- Precision: ~0.23–0.25  
- Recall: ~0.27–0.33  
- ROC-AUC: ~0.80  
- PR-AUC: ~0.17–0.18  

### **TabTransformer**  
*(Numbers depend on your upcoming overnight run)*

Historically:
- Higher recall  
- ROC-AUC in the mid-.70s to high-.70s  
- PR-AUC lower than the GNN  

Together, these models demonstrate **two fundamentally different strengths**:
- TabTransformer generalizes anomalies across categorical interactions.
- GNN isolates cleaner clusters of fraudulent behavior through graph structure.

---

# Future Work

Based on this semester’s experiments, the next steps for improving both models would be:

### **TabTransformer**
- Increase transformer depth (up to 6–8 layers)
- Add stochastic depth + stronger regularization
- Use embeddings tied across semantically related categories

### **GNN**
- Move to **PyTorch Geometric**
- Replace mean-GCN with GraphSAGE or GAT
- Expand graph construction using:
  - time-window co-occurrence
  - more identity relationships
  - behavior-based similarity edges
- Mini-batch sampling (GraphSAINT / Cluster-GCN)

These would push both models closer to Kaggle SOTA.

---

# Reproducibility — Setup Instructions

### 1. Create & activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Make sure the dataset is placed here:
```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

### 4. Train models
```
python run_tabTransformer.py
python run_gnn.py
```

---

# Final Remarks

This repository reflects my full end-to-end development of two deep architectures from scratch. I intentionally avoided pre-built frameworks (like PyTorch Geometric or TabNet) so I could understand the mechanics behind attention, graph message passing, and adaptive thresholding in imbalanced classification.

In my accompanying report and presentation, I walk through why each model was designed the way it was, how different architectural decisions affect performance, and how future iterations could move toward production-quality fraud detection systems.

---

If you have any questions about the implementation, model design choices, or future scalability considerations, feel free to reach out.