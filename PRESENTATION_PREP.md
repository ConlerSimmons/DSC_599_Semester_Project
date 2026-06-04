# DSC 599 — Presentation Preparation Guide
*Fraud Detection: TabTransformer vs Graph Neural Network*

---

## Table of Contents

1. [Project Overview — The Simple Version](#1-project-overview--the-simple-version)
2. [The Dataset — Everything You Need to Know](#2-the-dataset--everything-you-need-to-know)
3. [Why This Problem Is Hard](#3-why-this-problem-is-hard)
4. [Evaluation Metrics — Why Accuracy Is Wrong Here](#4-evaluation-metrics--why-accuracy-is-wrong-here)
5. [Feature Selection — How We Chose What to Feed the Models](#5-feature-selection--how-we-chose-what-to-feed-the-models)
6. [Model 1 — TabTransformer (Deep Dive)](#6-model-1--tabtransformer-deep-dive)
7. [Model 2 — Graph Neural Network (Deep Dive)](#7-model-2--graph-neural-network-deep-dive)
8. [Results and Comparison](#8-results-and-comparison)
9. [Key Design Decisions and Why You Made Them](#9-key-design-decisions-and-why-you-made-them)
10. [What to Talk About During the Presentation](#10-what-to-talk-about-during-the-presentation)
11. [Anticipated Questions and How to Answer Them](#11-anticipated-questions-and-how-to-answer-them)
12. [The Full Story — Beginning to End](#12-the-full-story--beginning-to-end)

---

## 1. Project Overview — The Simple Version

**One-sentence version:**
> I built two different kinds of neural networks from scratch to detect credit card fraud, and compared which one does a better job — and more importantly, *why.*

**Two-sentence version:**
> Credit card fraud detection is hard because fraudulent transactions are rare (about 1 in 28), and traditional models either miss too much fraud or flag too many legitimate transactions. I compared two modern deep learning architectures — one that uses attention mechanisms across transaction features, and one that builds a graph of related transactions — to understand which approach better captures fraud patterns and under what conditions.

**The actual research question:**
> *Do TabTransformer (attention over categorical features) and GNN (relational structure via shared identifiers) architectures capture patterns that traditional tabular methods miss — and if so, under what conditions?*

This is **comparative research**, not a competition. The goal is understanding *why* certain architectures work, not just *which* number is bigger.

---

## 2. The Dataset — Everything You Need to Know

### Source
The **IEEE-CIS Fraud Detection** dataset, originally released as a Kaggle competition by the IEEE Computational Intelligence Society and Vesta Corporation (a payment services company). This is real-world financial transaction data, not synthetic — which makes it meaningful.

### Scale
- **590,540 total transactions** spanning approximately 6 months
- **20,663 fraudulent transactions** — that's **3.5% of all transactions**
- **434 features** after merging both data files

### Two Files, One Dataset
The data comes in two files that must be joined:

| File | Rows | Columns | Contains |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | The transaction itself — amount, card info, address, engineered V-features |
| `train_identity.csv` | 144,233 | 41 | Device/identity info — only exists for ~24% of transactions |

They are joined on `TransactionID`. About **76% of transactions have no identity record**, so after the merge, most identity columns are NaN for the majority of rows. This is intentional and realistic — not every transaction has a captured device fingerprint.

### Feature Groups

| Prefix | What It Is | Type |
|---|---|---|
| `TransactionDT` | Seconds since a reference datetime (not a real timestamp) | Numeric |
| `TransactionAmt` | Dollar amount of the transaction | Numeric |
| `C1–C14` | Counting features (e.g., how many addresses are associated with this card) | Numeric |
| `D1–D15` | Timedelta features (e.g., days since the card was first seen) | Numeric |
| `V1–V339` | 339 anonymized Vesta-engineered features — their exact meaning is unknown | Numeric |
| `M1–M9` | Match features — binary yes/no checks (e.g., does billing address match shipping?) | Categorical |
| `card1–card6` | Card metadata (type, bank, etc.) | Categorical |
| `addr1, addr2` | Billing address info | Categorical |
| `P_emaildomain` | Purchaser email domain (gmail.com, yahoo.com, etc.) | Categorical |
| `R_emaildomain` | Recipient email domain | Categorical |
| `id_01–id_38` | Identity/device features from the identity table | Mixed |
| `DeviceType, DeviceInfo` | Device type (mobile/desktop) and specific device string | Categorical |

### The Missingness Problem
Many columns are **50–99% missing**. The V-columns especially have extreme sparsity. This is not a data quality issue — it reflects reality. Some fraud detection signals are only available for certain transaction types, devices, or merchants. A model has to be robust to this.

### Temporal Structure
`TransactionDT` is a timedelta (in seconds) from some reference point. Transactions are ordered in time. This matters for evaluation: **a model should be trained on past data and tested on future data**, just like a real production system. Models that peek at future data during training have inflated results that won't hold in deployment.

---

## 3. Why This Problem Is Hard

### Challenge 1: Extreme Class Imbalance
Only 3.5% of transactions are fraud. This sounds small, but it's actually worse than it sounds for a model:
- A completely useless model that always says "not fraud" achieves **96.5% accuracy**
- Standard loss functions (cross-entropy) treat every example equally — the model can minimize loss by mostly ignoring the minority class
- You need to explicitly re-weight the loss function to force the model to care about catching fraud

### Challenge 2: Anonymized Features
The 339 V-columns are Vesta's proprietary engineered features. Their names tell you nothing. You can't read "V189" and know what it means. The model has to figure out which of these 339 mystery columns are actually predictive — and most of them aren't.

### Challenge 3: High Cardinality Categoricals
`card1` alone has thousands of unique values. Standard one-hot encoding would create thousands of binary columns, most nearly empty. Embedding tables handle this gracefully — map each unique value to a dense vector of length 64.

### Challenge 4: Implicit Relationships
Two transactions from the same card aren't just similar — they're *related*. If transaction A on card X is fraud, that fact should raise suspicion about transaction B on card X. Traditional tabular models treat every row independently. A graph-based approach can model this dependency explicitly.

### Challenge 5: Temporal Drift
Fraud patterns evolve. Fraudsters adapt. A model trained on old data may not generalize to future fraud patterns. This is why temporal train/test splits are critical — a random split would allow the model to train on "future" data and look artificially better than it would in production.

---

## 4. Evaluation Metrics — Why Accuracy Is Wrong Here

This section is critical. If someone at your poster says "what accuracy did you get?", you need to be ready to redirect them.

### Why Accuracy Fails
At 3.5% fraud rate, accuracy is deceptive:
- A model that predicts "not fraud" for every single transaction: **96.5% accuracy**
- A model that catches 55% of fraud but flags some legitimate transactions: ~95% accuracy
- From accuracy alone, the useless model looks *better*

Accuracy measures *overall correctness* — but in fraud detection, we only care about performance on the minority class.

### The Metrics That Actually Matter

#### PR-AUC (Primary Metric)
**Precision-Recall Area Under Curve**

This is the most important metric in this project. Here's what it captures:

- **Precision** = Of all transactions you flagged as fraud, what fraction actually were? (controls false alarm rate)
- **Recall** = Of all actual fraudulent transactions, what fraction did you catch? (controls miss rate)

The PR curve plots precision vs. recall at every possible decision threshold. A perfect model has a curve that hugs the top-right corner (high precision AND high recall at all thresholds). A random model at 3.5% fraud rate has PR-AUC ≈ **0.035** — essentially a flat line near the bottom.

Both models scored around **0.34–0.44** on test PR-AUC. That is roughly **10–12× better than random**, which is meaningful performance.

#### ROC-AUC (Secondary Metric)
**Receiver Operating Characteristic — Area Under Curve**

Plots True Positive Rate vs. False Positive Rate across thresholds. A random model has ROC-AUC = 0.5. Both models scored ~0.83–0.85 here. ROC-AUC is reported because it's widely understood, but it can be misleadingly optimistic on imbalanced datasets.

#### Recall
What fraction of real frauds did the model catch?
- TabTransformer test: **54.95%** — catches a little over half of all fraud
- GNN test: **56.44%** — slightly higher, catches a bit more fraud

In a real fraud system, missing fraud has direct financial cost (chargebacks, liability). Higher recall = fewer missed frauds.

#### Precision
Of the transactions the model flags, what fraction are actually fraud?
- TabTransformer test: **31.90%**
- GNN test: **20.12%**

Low precision means lots of false alarms — real customers getting their cards blocked unnecessarily. There's a fundamental tradeoff between precision and recall; you can always catch more fraud by flagging more transactions, but then you also annoy more legitimate customers.

#### F1 Score
The harmonic mean of precision and recall. Balances both:
- TabTransformer test: **0.4036**
- GNN test: **0.2967**

#### F2 Score (used for threshold tuning)
Like F1 but weights recall **twice as heavily** as precision. This reflects the real-world judgment that missing fraud is worse than false alarms. Used to select the optimal operating threshold for each model.

#### Threshold
The models output a probability between 0 and 1. You need to pick a cutoff: above this number = flag as fraud. The choice changes the precision/recall tradeoff.
- TabTransformer threshold: **0.81** — only flags when it's very confident
- GNN threshold: **0.72** — flags at lower confidence

### The Generalization Gap
An important derived metric: **how much does performance drop from validation to test set?**
- TabTransformer: PR-AUC drops from 0.5385 (val) to 0.4402 (test) — a gap of **0.098**
- GNN: PR-AUC drops from 0.4147 (val) to 0.3437 (test) — a gap of **0.071**

The GNN generalizes more consistently despite lower absolute scores. This suggests the graph structure captures something about fraud patterns that is more stable across time.

---

## 5. Feature Selection — How We Chose What to Feed the Models

### The Problem With 434 Features
You can't just feed all 434 features to a neural network and hope for the best:
1. Most V-columns are nearly empty — they add noise, not signal
2. The network would have a huge input layer with mostly missing values
3. Training would be slower and the model harder to interpret

### The Solution: LightGBM Importance Ranking
Before training either deep learning model, we run a **100-tree LightGBM** on the training split. LightGBM (a fast gradient boosting framework) naturally ranks features by how often each one is used to make a split — this is "split importance."

This gives us a data-driven ranking of all 434 features. We then take:
- **Top 50 numeric features** by importance
- **Top 20 categorical features** by importance

**Total: 70 features fed to both models.**

Both the TabTransformer and GNN use **the exact same 70 features** — this is essential for a fair comparison. If one model had access to better features, the comparison would be invalid.

### Why LightGBM for Feature Selection?
- It handles missing values natively — no imputation needed before ranking
- It's fast (100 trees on 590k rows takes a few minutes)
- It implicitly captures non-linear importance — features that matter for complex interactions get credit
- It gives a baseline score "for free" — we can see what gradient boosting achieves on these features

---

## 6. Model 1 — TabTransformer (Deep Dive)

### The Core Idea
Transformers revolutionized NLP by treating words as tokens and learning which words attend to which other words. The key insight of TabTransformer is: **what if you treat tabular features the same way?**

Instead of a sentence like "the cat sat on the mat," you have a transaction: "card_type=Visa, email_domain=gmail.com, amount=142.50, device=mobile, ..." Each feature becomes a token. Attention learns which feature combinations signal fraud.

### Why This Makes Sense for Fraud
Traditional machine learning treats features independently by default — it doesn't automatically discover that "Visa card + gmail.com + mobile device + amount > $500 at 3am" is a suspicious combination. You'd have to engineer that interaction manually.

Transformer attention discovers these combinations automatically during training. If that particular pattern appears in fraudulent transactions, the attention mechanism will learn to weight those features highly when they co-occur.

### The Architecture — Step by Step

**Step 1: Feature Encoding**

Every feature, regardless of type, is converted into a 64-dimensional vector (a "token"):

- **Categorical features (20 columns):** Each unique value maps to a learned 64-dim embedding. For example, "gmail.com" in `P_emaildomain` maps to one vector, "yahoo.com" to another. The model learns these during training — similar email domains that behave similarly in the data will have similar embedding vectors.

- **Numeric features (50 columns):** Each numeric feature gets its own learned weight vector `w` and bias `b`. For feature `x_i`, the token is `x_i * w_i + b_i`. This is a **per-feature linear projection** — each of the 50 numeric features gets its own separate projection into 64 dimensions. This is different from collapsing all numerics into one token; individual projections let attention fire between specific numeric features.

**Step 2: Token Sequence**

After encoding, you have **70 tokens × 64 dimensions** = a matrix of shape [70, 64]. This is the "sentence" the transformer will read.

**Step 3: Transformer Encoder (3 layers)**

Each layer applies:
1. **Multi-head self-attention** (4 heads): Each token looks at all other tokens and decides how much to "attend" to each one. With 4 heads, the model can simultaneously learn 4 different types of feature relationships.
2. **Feed-forward network** (hidden dim = 256 = 4 × 64): Per-token transformation after attention.
3. **Dropout** (0.3): Randomly zeros out 30% of values during training to prevent overfitting.
4. **Residual connection + LayerNorm**: Standard transformer components that stabilize training.

The attention formula is: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
- Q, K, V are learned linear projections of the token sequence
- The softmax produces attention weights — how much each token focuses on each other token
- √d_k scaling prevents very large dot products that would make gradients vanish

**Step 4: Classification Head**

After 3 transformer layers, you have a 70 × 64 matrix of contextualized token representations. This is:
1. **Flattened** → vector of length 4480 (70 × 64)
2. **LayerNorm**
3. **Linear(4480 → 256) → ReLU → Dropout(0.2)**
4. **Linear(256 → 1)** → single logit

The logit is passed through sigmoid to get a fraud probability between 0 and 1.

### Training Details

| Setting | Value | Why |
|---|---|---|
| Batch size | 2048 | Large batches stabilize transformer training |
| Optimizer | AdamW | Adam + weight decay; standard for transformers |
| Learning rate | 2e-3 | Slightly aggressive; controlled by scheduler |
| Weight decay | 3e-4 | L2 regularization to prevent overfitting |
| LR warmup | 3 epochs linear | Prevents unstable gradients at the start |
| LR scheduler | ReduceLROnPlateau | Halves LR when val PR-AUC plateaus for 3 epochs |
| Loss | BCEWithLogitsLoss | Binary cross-entropy for 0/1 labels |
| pos_weight | ~27× | Upweights fraud examples to compensate for class imbalance |
| Label smoothing | 0.05 | Soft targets (0.05/0.95 instead of 0/1) reduce overconfidence |
| Early stopping | patience=10 | Stop if val PR-AUC doesn't improve for 10 epochs |
| Max epochs | 50 | Upper bound |
| Threshold tuning | F2 score on val set | Finds optimal operating threshold |

**pos_weight explained:** The loss function treats each fraud example as if it appeared 27 times. This is the approximate ratio of non-fraud to fraud in the training data (96.5% / 3.5% ≈ 27.6). Without this, the model can minimize loss by mostly ignoring fraud.

### TabTransformer Performance

| Metric | Validation | Test |
|---|---|---|
| PR-AUC | 0.5385 | **0.4402** |
| ROC-AUC | 0.8817 | 0.8471 |
| F1 | 0.4606 | 0.4036 |
| Recall | 0.6072 | 0.5495 |
| Precision | 0.3710 | 0.3190 |
| Threshold | 0.8113 | — |

The 0.81 threshold means the model is conservative — it only flags a transaction when it's at least 81% confident it's fraud. Despite this conservatism, it still catches 55% of fraud in the test set.

---

## 7. Model 2 — Graph Neural Network (Deep Dive)

### The Core Idea
Tabular models treat every transaction as an isolated row. But transactions are not isolated — they're connected. Two transactions from the same card, same device, or same email address are *related*. If one of them is fraud, that's a signal about the other.

A Graph Neural Network models these connections explicitly. Each transaction is a node. Edges connect transactions that share an identity attribute. During training, each node aggregates information from its neighbors — fraud signals "spread" across connected transactions.

**The fraud hypothesis:** Fraudsters reuse payment instruments. Someone who steals a card doesn't use it once. They make multiple transactions, possibly on different merchants, before the card is blocked. If any of those transactions is confirmed fraud, the graph structure means all connected transactions get a raised suspicion score.

### Graph Construction

**What qualifies as an edge?**
Two transactions are connected if they share the same value in any of these columns:
- `card1`, `card4`, `card6` — card number / type / bank
- `addr1` — billing address
- `P_emaildomain`, `R_emaildomain` — purchaser and recipient email domains
- `id_30`, `id_31` — OS and browser info
- `DeviceInfo` — specific device fingerprint

**Why these columns specifically?** These are the identity attributes that a fraudster would have to reuse across transactions. A fraudster can't easily change their card number, device fingerprint, or email address between transactions. Address and email domain are stickier than, say, transaction amount.

**NaN handling:** If two transactions both have NaN for `card1`, that does NOT create an edge. NaN means "we don't know the card number" — it doesn't mean they share the same card. Connecting NaN groups would add massive amounts of meaningless edges.

**Group size handling:**
Naively connecting all transactions in a group creates a clique — if 1000 transactions share the same card, you'd have 1000 × 999 / 2 ≈ 500,000 edges just for that one card. This is quadratic growth.

| Group Size | Connection Strategy | Why |
|---|---|---|
| 2–10 transactions | Full clique (everyone ↔ everyone) | Small enough that dense connections are fine |
| 11–1,000 transactions | Hub-and-spoke + sequential chain | Linear edge count instead of quadratic |
| 1,000+ transactions | Skip entirely | Too large — likely generic shared attributes (e.g., "gmail.com" shared by millions), not meaningful identity links |

Self-loops are added to every node so each transaction aggregates its own features alongside its neighbors' features during message passing.

**Why identity edges instead of k-NN edges?**
An earlier design used k-nearest-neighbor edges — connect the 5 most numerically similar transactions in feature space. This was removed because:
1. Computing k-NN on 590,000 × 50 features is O(N²) — building the graph would take hours
2. Numerical similarity doesn't encode the fraud hypothesis — two transactions can be numerically similar without having any meaningful relationship
3. Identity edges are O(N) to compute and directly encode the "shared payment instrument" hypothesis

### Architecture — Step by Step

**Step 1: Node Feature Encoding**
Same as TabTransformer — each node (transaction) gets encoded into a 256-dimensional representation:
- Numeric features → Linear(50, 32) → ReLU → 32-dim vector
- Each categorical feature → Embedding(vocab_size + 1, 32) → 32-dim vector
- Concatenate all embeddings → Linear(32 × 21, 256) → ReLU → **256-dim node representation**

**Step 2: Message Passing (3 residual layers)**
This is where the GNN differs from TabTransformer. For each node `v` at each layer:

1. **Aggregate:** Compute the mean embedding of all neighbors: `h_agg = mean({h_u : u ∈ N(v)})`
2. **Transform:** `h_new = ReLU(W · h_agg)` where W is a learned weight matrix
3. **Dropout:** Randomly zero out activations during training
4. **Residual + LayerNorm:** `h = LayerNorm(h_new + h_prev)`

The residual connection (`h_new + h_prev`) is critical. Without it, after 3 layers of mean aggregation, all nodes in a densely connected component would converge to the same vector — all their representations would become identical. This is called **over-smoothing** and it defeats the purpose of the GNN. The residual connection preserves the node's own identity throughout message passing.

**After 3 layers:** each node's 256-dim representation encodes both its own features and information from up to 3 hops away in the graph.

**Step 3: Classification**
- Linear(256 → 1) → logit → sigmoid → fraud probability

### Training Details

| Setting | Value | Why |
|---|---|---|
| Training mode | Full-graph | Message passing requires all neighbors simultaneously |
| Optimizer | AdamW, lr=1e-3, wd=1e-4 | Standard |
| Loss | BCEWithLogitsLoss + pos_weight (~27×) | Same imbalance correction as TabTransformer |
| Label smoothing | 0.05 | Same as TabTransformer |
| Early stopping | patience=20 | GNNs converge slowly; 10 epochs isn't enough |
| Max epochs | 100 | GNNs need more epochs than mini-batch models |
| Threshold tuning | F2 score on val set | Same approach as TabTransformer |
| Device | CUDA GPU | Full-graph on 590k nodes was OOM on Apple Silicon |

**Why full-graph training?** In mini-batch training, you sample a random subset of nodes per batch. But message passing requires knowing each node's neighbors — which may not be in the batch. Full-graph training loads all 590k nodes and all edges into GPU memory and does one gradient step per epoch. This is memory-intensive but conceptually simple.

### GNN Performance

| Metric | Validation | Test |
|---|---|---|
| PR-AUC | 0.4147 | **0.3437** |
| ROC-AUC | 0.8528 | 0.8317 |
| F1 | 0.3456 | 0.2967 |
| Recall | 0.5631 | **0.5644** |
| Precision | 0.2493 | 0.2012 |
| Threshold | 0.7228 | — |

The 0.72 threshold means the GNN is more trigger-happy — it flags transactions at 72% confidence vs. TabTransformer's 81%. This contributes to higher recall but lower precision.

---

## 8. Results and Comparison

### Head-to-Head

| Metric | TabTransformer | GNN | Winner |
|---|---|---|---|
| test PR-AUC ★ | **0.4402** | 0.3437 | TabTransformer |
| test ROC-AUC | **0.8471** | 0.8317 | TabTransformer |
| test Recall | 0.5495 | **0.5644** | GNN |
| test Precision | **0.3190** | 0.2012 | TabTransformer |
| test F1 | **0.4036** | 0.2967 | TabTransformer |
| Val→Test Gap (PR-AUC) ↓ | 0.098 | **0.071** | GNN |

★ Primary metric — ↓ lower is better for the gap

### What This Means

**TabTransformer is the stronger model overall.** It wins on PR-AUC (the primary metric), ROC-AUC, precision, and F1. When it flags something as fraud, it's right more often. It's a more precise instrument.

**GNN catches marginally more fraud.** Its recall is 56.4% vs. TabTransformer's 54.9% — about 1.5 percentage points better. But this comes at a steep cost: precision drops from 31.9% to 20.1%, meaning 80% of its fraud flags are false alarms vs. 68% for TabTransformer.

**GNN generalizes more consistently.** The val→test PR-AUC gap (how much performance degrades from validation to test) is 0.071 for GNN vs. 0.098 for TabTransformer. The graph-based relational structure may encode fraud patterns that are more stable across time, even if absolute performance is lower.

**Both models are meaningful.** Random PR-AUC at 3.5% fraud rate ≈ 0.035. TabTransformer achieves 0.44 — roughly **12.6× better than random**. GNN achieves 0.34 — roughly **9.8× better than random**. Both represent real signal.

### The Complementarity Point
The two models fail differently. TabTransformer misses fraud that lacks suspicious feature combinations but is connected to other known-fraud transactions. GNN misses fraud from new actors with no graph connections. An ensemble of both models would likely outperform either alone — a natural extension of this work.

### Why TabTransformer Wins Here
This dataset is feature-rich but graph-sparse. Many transactions have no identity data at all (76% have no identity record), so 76% of nodes have only their transaction features — they receive no benefit from graph structure because they have no edges. TabTransformer can extract value from every single transaction's features. GNN can only add value for the subset with graph connections.

---

## 9. Key Design Decisions and Why You Made Them

These are the decisions that show depth of understanding. Know all of these cold.

### "Why PR-AUC as the primary metric?"
Accuracy at 3.5% fraud rate is useless — a constant "not fraud" classifier gets 96.5%. PR-AUC measures the precision-recall tradeoff across all thresholds and is robust to class imbalance. It penalizes models that only do well on the majority class. ROC-AUC is also robust to imbalance but is known to be overly optimistic — PR-AUC is more conservative and more actionable.

### "Why F2 for threshold tuning?"
F1 treats precision and recall equally. In fraud detection, missing a fraud costs more (direct financial loss) than a false alarm (customer inconvenience). F2 weights recall 2× over precision, biasing the threshold selection toward catching more fraud. This is a business judgment embedded in the metric.

### "Why temporal splits?"
A random 70/15/15 split would allow training data to include transactions from *after* the validation data. In production, a model only sees past data. Testing on randomly mixed future/past data overestimates real-world performance. Temporal splits are the only honest evaluation.

### "Why the same features for both models?"
Fair comparison. If one model had access to more informative features, any performance difference could be attributed to features rather than architecture. Holding features constant isolates the architectural effect.

### "Why LightGBM for feature selection instead of just picking manually?"
339 V-columns are anonymized — there's no way to know which ones matter without data-driven ranking. LightGBM's split importance is a principled, interpretable method for ranking features on the actual target variable. Manual selection would be arbitrary.

### "Why identity edges instead of k-NN edges for the GNN?"
k-NN on 590k × 50 features is O(N²) — building the graph would take hours and produce edges based on numerical similarity, not the fraud hypothesis. Identity edges are O(N), directly encode the "shared payment instrument" fraud hypothesis, and are the domain-meaningful connection.

### "Why full-graph training for the GNN?"
Message passing is inherently global — each node's representation depends on its neighbors, which depend on *their* neighbors. Mini-batching a graph requires graph partitioning (neighborhood sampling), which introduces approximation error and requires significant code complexity. Full-graph training is exact and simpler, and works fine with enough GPU memory.

### "Why residual connections in the GNN?"
Without residual connections, mean aggregation over multiple layers causes over-smoothing — all nodes in a connected component converge to the same representation, collapsing all individual identity. Residuals preserve each node's own features throughout message passing.

### "Why label smoothing?"
Hard labels (0 and 1) can cause overconfidence. With 27× pos_weight, the model might learn to output extremely high scores for fraud examples just to minimize loss. Label smoothing softens targets to 0.05/0.95 — the model is penalized for being *too* confident, which improves generalization.

### "Why different training lengths? (TabTransformer: 50 max, GNN: 100 max)"
Mini-batch training (TabTransformer) sees the full dataset many times per epoch — convergence is faster. Full-graph training (GNN) does one gradient step per epoch — it needs more epochs to converge. 100 epoch max for GNN with patience=20 mirrors the effective number of gradient steps.

### "Why not just use XGBoost/LightGBM as the final model?"
The research question isn't "what's the best model for fraud detection?" It's "what do attention and graph structure add beyond standard approaches?" XGBoost and LightGBM serve as baselines to establish the floor — the deep learning models are investigated for what *additional* patterns they capture and why.

---

## 10. What to Talk About During the Presentation

### Opening Hook (30 seconds)
Start with a relatable framing: "Imagine you're a bank. Every day, hundreds of thousands of transactions flow through your system. 3.5% of them are fraud — but you don't know which 3.5%. You could block every suspicious transaction, but then you'd also block 68% of legitimate ones. The question I investigated is: can modern AI architectures help us be smarter about this tradeoff?"

### Core Narrative Arc
1. **Problem framing** — fraud detection is hard because of imbalance, anonymized features, and implicit relationships
2. **Two hypotheses** — attention discovers feature interactions; graphs propagate fraud signals through connected transactions
3. **Fair comparison setup** — same data, same features, same evaluation protocol
4. **Results** — TabTransformer wins overall; GNN wins on recall and generalization consistency
5. **Insight** — the two architectures are complementary; an ensemble is the natural next step

### What to Emphasize at the Poster
- The **PR-AUC metric** — explain it proactively, don't wait for someone to ask
- The **10-12× better than random** framing — this contextualizes the numbers
- The **graph diagram** — walk through it: "each circle is a transaction, edges connect transactions sharing the same card or device, and fraud probability spreads through those connections"
- The **architecture tradeoff** — TabTransformer is more precise, GNN catches more fraud but casts a wider net. Neither is universally better — it depends on whether missing fraud or annoying customers is more costly

### Things Not to Overemphasize
- Don't say "accuracy" — redirect to PR-AUC immediately
- Don't oversell the numbers — 0.44 PR-AUC is good but not deployment-ready; be honest about that
- Don't say "my GNN" is worse — frame it as "complementary strengths"

---

## 11. Anticipated Questions and How to Answer Them

### "What accuracy did you get?"
"Accuracy isn't a useful metric here because the dataset is highly imbalanced — 96.5% of transactions are legitimate, so a model that never predicts fraud would score 96.5% accuracy without catching anything. I use PR-AUC as the primary metric, which measures the tradeoff between precision and recall at all thresholds. Both models achieve around 10–12× better than a random baseline."

### "Which model is better?"
"TabTransformer wins on most metrics — PR-AUC of 0.44 vs. 0.34 for the GNN, and better precision and F1. But the GNN has the highest recall, meaning it catches slightly more fraud — 56.4% vs. 54.9%. The GNN also generalizes more consistently across time. So it depends on your priorities: if you need high precision, TabTransformer. If recall is paramount, GNN. Ideally, you'd ensemble them."

### "Why not just use XGBoost?"
"We do run XGBoost and LightGBM as baselines — this is comparative research, not a claim that deep learning always wins. The research question is specifically about whether attention over features and graph-based relational reasoning capture patterns that traditional methods miss. The architectural inductive biases are the thing under study."

### "Why didn't you get higher scores?"
"A few honest reasons. First, 76% of transactions have no identity records, so most GNN nodes have no edges — the graph approach is limited by data sparsity. Second, 339 of the features are fully anonymized with unknown semantics, limiting interpretability. Third, both models have a non-trivial val→test gap on a temporal split — this is real generalization challenge, not a bug. The scores we achieve are competitive with published literature on this dataset at a similar feature budget."

### "What is a transformer doing here? I thought those were for text."
"Transformers were originally designed for text, but the underlying mechanism — self-attention over a sequence of tokens — is general. In NLP, each word is a token. Here, each feature is a token. The attention mechanism learns which features 'pay attention to' which other features. In the same way language models learn that 'not' and 'good' should interact, TabTransformer learns that 'card_type=prepaid' and 'new email domain' are suspicious together."

### "What's the difference between a GNN and a regular neural network?"
"A regular neural network treats every input row independently — it processes one transaction at a time with no knowledge of other transactions. A GNN adds a graph structure where nodes are transactions and edges represent relationships. During forward pass, each node doesn't just use its own features — it aggregates information from its neighbors. So a suspicious node's signal spreads to its neighbors through message passing. This lets the model capture relational patterns that row-independent models can't."

### "Why did you make the graph the way you did?"
"The connections are based on shared identity attributes — same card number, same email domain, same device fingerprint. The fraud hypothesis is that fraudsters reuse payment instruments. If I see that card X was used fraudulently on Monday, and that same card X appears in a Tuesday transaction, that's a strong signal. I deliberately skipped k-nearest-neighbor edges, which would connect numerically similar transactions — that's computationally expensive and doesn't encode the fraud hypothesis as directly."

### "How would this work in a real production system?"
"A few things would need to change. First, you'd need to update the graph incrementally as new transactions arrive — full-graph recomputation every transaction isn't practical. Second, the threshold would need to be calibrated against actual business costs — the F2 threshold we chose is one policy, but different banks have different tolerance for false alarms vs. missed fraud. Third, you'd want a monitoring system to detect when fraud patterns drift and trigger retraining."

### "Could you combine both models?"
"Yes, and that's a natural extension. The models capture different signals — TabTransformer captures feature interactions, GNN captures relational structure. An ensemble (e.g., averaging their probability outputs) would likely outperform either alone, especially for transactions with rich graph connections. This is actually mentioned in the research conclusion."

### "What does 'attention' mean technically?"
"Attention is a mechanism that computes a weighted sum of values, where the weights are determined by compatibility between a query and a set of keys. In practice: for each feature token, attention asks 'which other feature tokens are most relevant to me right now?' and assigns higher weight to those. The formula is softmax(QK^T / √d) × V. The softmax ensures weights sum to 1. The √d scaling prevents very large dot products that would push softmax into regions with vanishing gradients."

### "Why did you build these from scratch instead of using existing implementations?"
"Two reasons. First, understanding — when you implement every layer yourself, you genuinely understand what the model is doing. A black-box import doesn't give you the same depth. Second, control — I needed to make specific architectural choices (per-feature numeric projections, residual GNN layers, full-graph training) that off-the-shelf implementations don't always support cleanly. The research insights come from those choices."

### "What would you do differently?"
"A few things. First, implement graph mini-batching (NeighborSampler) so the GNN could train faster without requiring a full GPU forward pass. Second, investigate whether ensemble methods would outperform either model alone. Third, try to explain which feature interactions the TabTransformer is actually attending to — attention visualization would make the model more interpretable. Fourth, experiment with heterogeneous graph structures that treat different edge types (card edges vs. device edges) differently."

### "What is the real fraud rate in practice?"
"The IEEE-CIS dataset has 3.5% fraud, which is actually higher than many real-world scenarios — some card networks report rates below 0.1%. At lower fraud rates, PR-AUC degrades and the precision-recall tradeoff becomes even more severe. The models here would perform differently at 0.1% fraud. However, the architectural comparison — which model captures which patterns — would likely hold."

### "Why the tabular approach? Why not just use image or sequence models?"
"Transaction data doesn't have the natural sequential or spatial structure that makes images or sequences well-suited to CNNs or RNNs. Each transaction is a row of mixed-type features — some categorical, some numeric, with lots of missing values. TabTransformer is specifically designed for this tabular structure. The transformer architecture is general enough to adapt to any token representation, including feature tokens."

---

## 12. The Full Story — Beginning to End

*This section explains the entire project from scratch — start here if someone has no ML background, and add depth as needed.*

---

### Part 1: The Problem

Every time you swipe a credit card, a fraud detection system has milliseconds to decide whether to approve or decline the transaction. Banks process hundreds of millions of transactions per day globally. A small fraction — around 3 to 4 percent — are fraudulent.

This sounds manageable. 3.5% is small. But consider what it means in practice:
- 590,000 transactions analyzed in this dataset → about 21,000 are fraud
- If you miss even 45% of fraud, you're letting 9,000 fraudulent transactions through
- If you over-flag, you might block 50,000 legitimate transactions — customers who just lose access to their money

The challenge isn't just binary classification. It's **operating at a specific tradeoff point** between catching fraud and not annoying legitimate customers.

---

### Part 2: The Data

The IEEE-CIS dataset is real transaction data from Vesta Corporation, a payment processing company. It covers six months of transactions from a real payment network.

The data has a fundamental structure:

**Transaction table:** Every transaction ever processed. Information includes the dollar amount, some card metadata (type, bank, etc.), billing address, email domains used, and 339 anonymized features labeled V1 through V339. These V-features are Vesta's proprietary fraud signals — their exact meaning is trade secret.

**Identity table:** Device and browser information for about 24% of transactions — the ones where Vesta could capture a device fingerprint. This includes the OS, browser, device model, screen resolution, and other technical details.

These are joined together. The 76% of transactions with no identity record just have blank (NaN) values for those columns.

After joining: 590,540 rows × 434 columns. That's a lot of data with a lot of missing values.

---

### Part 3: Why Standard Approaches Struggle

**Neural networks** default to treating every row independently. Each transaction is processed in isolation, with no knowledge that other transactions exist. This misses a key fraud signal: fraudsters reuse their tools. The same card appears multiple times. The same device shows up again and again.

**Traditional ML** (decision trees, random forests) handles tabular data well, but doesn't naturally model multi-way feature interactions. It might learn "amount > $300 → suspicious" but struggle with "amount > $300 AND email is new AND device is mobile → very suspicious as a combination."

**Both approaches** suffer from the class imbalance. 96.5% of examples are "not fraud." A model can easily achieve high accuracy by just learning "always say not fraud." The loss function needs to be explicitly adjusted to force the model to care about the minority class.

---

### Part 4: The Two Architectural Hypotheses

This research asks: are there specific architectural designs that address these challenges better?

**Hypothesis 1: Attention over features helps**
If we treat every feature as a token — like a word in a sentence — and run transformer attention across those tokens, the model can learn which feature combinations are suspicious. Card type, email domain, device type, and transaction amount aren't just individual signals; their interactions matter. Attention can discover these combinations without needing manual feature engineering.

This is the **TabTransformer** approach.

**Hypothesis 2: Graph structure helps**
If we explicitly represent the connections between transactions — same card, same email, same device — and let fraud signals propagate through those connections, the model can catch fraud that's connected to other known-fraud transactions, even if the individual transaction features look innocuous.

This is the **Graph Neural Network** approach.

---

### Part 5: Feature Selection — Picking the Right 70

With 434 features, most of which are either nearly empty or meaningless, we can't just feed everything to the models. We need to pick the most informative ones.

The approach: run a LightGBM model (a fast, high-performance decision tree ensemble) on the training data and ask it: which features did you use most? Features that get used a lot in tree splits are presumably informative. Features that never get used are probably noise.

This gives us a ranked list of all 434 features. We take the top 50 numeric and top 20 categorical — 70 features total. Both models use this same set, which is crucial for a fair comparison.

---

### Part 6: Training the TabTransformer

The TabTransformer takes the 70 features and encodes each one as a 64-dimensional vector. You now have a 70×64 matrix — 70 tokens, each represented by 64 numbers.

This matrix is fed into a 3-layer transformer encoder. Each layer runs self-attention: every token looks at every other token and computes how much to "pay attention" to it. The attention weights are learned during training — the model discovers which feature combinations are important for fraud detection.

After 3 transformer layers, the contextualized token representations are flattened into a long vector and passed through a small neural network (the "head") to produce a single output: the fraud probability.

Training uses mini-batches of 2048 transactions at a time. Each batch, the model makes predictions, compares them to the true labels, computes a loss value, and updates its weights via backpropagation. The loss function is adjusted to treat each fraud example as if it appeared 27 times (counteracting the imbalance). The model trains for up to 50 epochs with early stopping.

At the end of training, we pick a decision threshold: above what probability do we flag as fraud? We pick the threshold that maximizes F2 score (which weights recall more than precision) on the validation set.

---

### Part 7: Building the Graph for the GNN

Before training the GNN, we have to build the graph. Each of the 590,000 transactions becomes a node. We add edges based on shared identity attributes.

The process:
1. Group transactions by `card1` value. If two transactions share the same card number, add an edge between them.
2. Repeat for `card4`, `card6`, `addr1`, `P_emaildomain`, `R_emaildomain`, `id_30`, `id_31`, `DeviceInfo`.
3. For very large groups (e.g., everyone using gmail.com — millions of transactions), skip them — they're too large to be meaningful connections.
4. Add self-loops (each node connects to itself) so during aggregation, each node includes its own features.

The result: a graph with 590,000 nodes and millions of edges. This is the structure the GNN will learn on.

---

### Part 8: Training the GNN

The GNN starts by encoding each node's features the same way as TabTransformer — numeric projection + categorical embeddings, concatenated into a 256-dim node representation.

Then comes the GNN-specific part: **message passing**. Three rounds:

**Round 1:** Each node looks at all its neighbors and computes the average of their 256-dim representations. This average is transformed by a learned weight matrix. The result is combined with the node's own representation (residual connection) and normalized.

**Round 2:** Same thing, but now each node's representation already includes information from its 1-hop neighbors. So round 2 effectively aggregates 2-hop information.

**Round 3:** Now 3-hop information is incorporated.

After 3 rounds, a node representing a legitimate transaction that happens to share a card with a fraudulent transaction will have its representation influenced by that fraud signal. That's the GNN's advantage.

The full graph (all 590k nodes, all edges) is loaded into GPU memory at once. One forward pass processes all nodes simultaneously. Training runs for up to 100 epochs with early stopping.

---

### Part 9: Results

After training both models on the first 70% of the dataset (ordered by time) and tuning thresholds on the middle 15%, we evaluate on the final 15% — data the models have never seen, from later in time.

**TabTransformer:**
- Catches **54.95%** of fraud in the test set
- When it flags something, it's right **31.9%** of the time
- PR-AUC: **0.4402** — about 12.6× better than random

**GNN:**
- Catches **56.44%** of fraud in the test set
- When it flags something, it's right **20.1%** of the time
- PR-AUC: **0.3437** — about 9.8× better than random

**Takeaway:** TabTransformer is more precise. GNN catches slightly more fraud but casts a much wider net. TabTransformer wins on the primary metric (PR-AUC). GNN wins on recall and generalizes more consistently across time.

Neither model is deployment-ready on its own — real fraud systems use ensembles, rule-based post-processing, and continuous retraining. But as a research comparison, both demonstrate meaningful signal well above chance.

---

### Part 10: The Research Conclusion

For this dataset, **feature-level attention (TabTransformer) outperforms relational graph reasoning (GNN)** on the primary metric.

**Why?** The GNN is limited by graph sparsity. 76% of transactions have no identity data and therefore no graph edges. These nodes receive no benefit from message passing — they're isolated in the graph and their representation comes entirely from their own features. TabTransformer, by contrast, extracts value from every transaction's features regardless of connectivity.

The GNN's strength — propagating fraud signals through connected transactions — only manifests for the 24% of transactions with identity data and meaningful graph connections. For those transactions, the GNN likely performs better than TabTransformer. For the 76% majority, it doesn't add anything.

**The complementarity insight:** An ensemble would likely outperform either model alone. TabTransformer handles feature interactions; GNN handles relational propagation. Together, they cover the full space of fraud signals in this dataset.

This is the real research contribution: not just which number is bigger, but *why* each architecture succeeds or fails under the specific constraints of this data — extreme imbalance, anonymized features, sparse identity information, and temporal structure.

---

*Document prepared for DSC 599 poster presentation. All results from the `colab` branch.*
*Last updated: April 2026*
