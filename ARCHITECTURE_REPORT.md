# Model Architecture & Performance Report
*DSC 599 Capstone — IEEE-CIS Fraud Detection*

---

## Dataset

I'm working with the IEEE-CIS Fraud Detection dataset from Kaggle. It contains 590,540 financial transactions over roughly six months, with 3.5% of them labelled as fraudulent. The data comes in two files — `train_transaction.csv` and `train_identity.csv` — which I join on `TransactionID`. About 76% of transactions have no identity record, so those rows are NaN-heavy after the merge.

The class imbalance is severe enough that accuracy is a meaningless metric here. A model that never predicts fraud would score 96.5%. I track **PR-AUC** as the primary metric throughout, supported by recall, precision, F1, and ROC-AUC.

**Split:** 70% train / 15% val / 15% test, sorted by `TransactionDT` so the model always trains on older data and evaluates on newer data.

---

## Feature Selection

Rather than heuristically grabbing the first N columns, I run a 100-tree LightGBM on the training split and rank every candidate feature by split importance. This matters because the dataset has 339 anonymised V-columns — their names tell me nothing, so I need the model to tell me which ones are actually useful. Both the TabTransformer and GNN use the same 50 numeric + 20 categorical features selected this way, keeping the comparison fair.

---

## Model 1 — TabTransformer

### Architecture

The core idea is to treat every feature — both categorical and numeric — as a token, then apply transformer attention across the full token sequence. This lets the model discover which feature combinations matter for a given transaction without me having to engineer those interactions by hand.

**Input encoding:**
- Each categorical column gets its own embedding table (vocab_size × 64). So `card_type`, `email_domain`, `device_type` etc. all live in the same 64-dimensional space and can be directly compared by attention.
- Each numeric feature gets its own weight vector and bias vector (shape: 64). The forward pass is `token_i = x_i * w_i + b_i` — a per-feature linear projection. This was a deliberate choice over collapsing all numerics into one token; it lets attention fire between individual numeric features too.

**Transformer:**
- 3 TransformerEncoder layers, 4 attention heads each
- Feed-forward dim = 4 × d_token = 256
- Dropout = 0.2 (increased from 0.1 to reduce overfitting after observing a large val→test gap)

**Head:**
- LayerNorm → Linear(n_tokens × 64, 256) → ReLU → Dropout(0.2) → Linear(256, 1)
- Output is a single logit; sigmoid gives the fraud probability

**Total token count:** 50 numeric + 20 categorical = 70 tokens per transaction

### Training

- **Batch size:** 2048 (scaled up from 512 — GPU has plenty of VRAM)
- **Optimizer:** AdamW, lr=2e-3, weight_decay=3e-4
- **LR schedule:** 3-epoch linear warmup, then ReduceLROnPlateau (factor=0.5, patience=3)
- **Loss:** BCEWithLogitsLoss with pos_weight (~27× — the neg/pos ratio in training)
- **Label smoothing:** 0.05 — softens targets from 0/1 to 0.05/0.95 to reduce overconfidence
- **Early stopping:** patience=10 on val PR-AUC
- **Max epochs:** 50
- **Threshold:** tuned on val set using F2 score (weights recall 2× over precision)

### Performance

| Metric | Validation | Test |
|---|---|---|
| PR-AUC | 0.5385 | **0.4402** |
| ROC-AUC | 0.8817 | 0.8471 |
| F1 | 0.4606 | 0.4036 |
| Recall | 0.6072 | 0.5495 |
| Precision | 0.3710 | 0.3190 |
| Threshold | 0.8113 | — |

**Key observations:**
- Catches 55% of fraud in the test set
- Val→test PR-AUC gap of 0.098 — narrowed from 0.109 after increasing dropout to 0.3 and weight_decay to 1e-3
- Threshold of 0.81 — the model is confident before flagging
- Wins on PR-AUC, F1, and precision compared to the GNN

---

## Model 2 — Graph Neural Network (GNN)

### Graph Construction

Every transaction is a node. I connect two nodes with an edge if they share a value in any of these identity columns:

`card1`, `card4`, `card6`, `addr1`, `P_emaildomain`, `R_emaildomain`, `id_30`, `id_31`, `DeviceInfo`

The hypothesis is that fraudsters reuse payment instruments. If transaction A and B used the same card, they're probably from the same person — and if one is fraud, that's a signal about the other.

I skip NaN groups deliberately. If two transactions both lack a card number, that doesn't make them related — connecting them would add noise, not signal.

**Group handling:**
- Groups of 2–10 transactions: full clique (everyone connected to everyone)
- Groups of 11–1000: hub-and-spoke with a sequential chain — linear edge count instead of quadratic
- Groups over 1000: skipped entirely (too large, likely noise)

Self-loops are added so each node aggregates its own features alongside its neighbours.

I tried k-NN edges initially (connecting numerically similar transactions) but removed them — computing nearest neighbours on 590k × 50 features is O(N²) and would take hours just to build the graph. Identity edges are O(N) and more directly encode the fraud hypothesis.

### Architecture

**Input encoding:**
- Numeric features → Linear(50, 32) → ReLU → 32-dim embedding
- Each categorical column → Embedding(vocab_size + 1, 32)
- All embeddings concatenated → Linear(32 × (1 + n_cat), 256) → ReLU → 256-dim node representation

**Message passing (3 residual layers):**

For each layer:
1. Mean-aggregate neighbour features: `h_agg = mean({h_u : u ∈ N(v)})`
2. Transform: `h_new = ReLU(W · h_agg)`
3. Dropout
4. Residual + LayerNorm: `h = LayerNorm(h_new + h_prev)`

The residual connections prevent over-smoothing — without them, deep GNNs tend to make all node representations converge to the same value, which defeats the purpose.

**Output:**
- Linear(256, 1) → single logit per node → sigmoid → fraud probability

**hidden_dim:** 256

### Training

- **Mode:** Full-graph (all 590k nodes + all edges in every forward pass — no mini-batching)
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Loss:** BCEWithLogitsLoss with pos_weight (~27×)
- **Label smoothing:** 0.05
- **Early stopping:** patience=20 (GNNs converge more slowly than mini-batch models)
- **Max epochs:** 100
- **Threshold:** tuned on val set using F2 score
- **Device:** CUDA (full-graph training was OOM on MPS/Apple Silicon)

### Performance

| Metric | Validation | Test |
|---|---|---|
| PR-AUC | 0.4147 | **0.3437** |
| ROC-AUC | 0.8528 | 0.8317 |
| F1 | 0.3456 | 0.2967 |
| Recall | 0.5631 | **0.5644** |
| Precision | 0.2493 | 0.2012 |
| Threshold | 0.7228 | — |

**Key observations:**
- Highest recall of either model — catches 56% of fraud in the test set
- Lower precision — casts a wider net, more false alarms
- Threshold of 0.72 — less confident than TabTransformer before flagging
- Val→test gap of 0.071 — reasonable generalisation on a temporal split

---

## Head-to-Head Comparison

| Metric | TabTransformer | GNN | Winner |
|---|---|---|---|
| test PR-AUC | **0.4402** | 0.3437 | TabTransformer |
| test ROC-AUC | **0.8471** | 0.8317 | TabTransformer |
| test Recall | 0.5495 | **0.5644** | GNN |
| test Precision | **0.3190** | 0.2012 | TabTransformer |
| test F1 | **0.4036** | 0.2967 | TabTransformer |
| Val→Test gap (PR-AUC) | 0.098 | **0.071** | GNN |

**Summary:**
TabTransformer is the stronger model overall — it wins on PR-AUC, ROC-AUC, F1, and precision. The GNN catches marginally more fraud (56% vs 55% recall) but at the cost of significantly more false positives. The GNN's smaller val→test gap (0.071 vs 0.098) suggests the relational graph structure generalises more consistently across time, even though it doesn't win on the primary metric.

Both models perform roughly 10–12× better than random (random PR-AUC ≈ 0.035 at 3.5% fraud rate).

---

## What Each Architecture Is Good At

**TabTransformer:** learns which combinations of categorical features are suspicious. Card type × email domain × device type — the attention mechanism discovers these interactions without being told they matter.

**GNN:** propagates fraud signals across connected transactions. If a card was used fraudulently once, every other transaction on that card gets a higher suspicion score through message passing. This is relational reasoning that the TabTransformer can't do.

**Research conclusion:** For this dataset, feature-level attention (TabTransformer) outperforms relational graph reasoning (GNN) on the primary metric. However, the two approaches are complementary — an ensemble would likely outperform either alone.
