# Instacart Two-Tower Recommender System

A production-style two-stage retrieval + reranking pipeline built on the [Instacart Online Grocery dataset](https://www.kaggle.com/c/instacart-market-basket-analysis).

---

## Architecture

### System Overview

```
                        ┌─────────────────────────────────┐
                        │          TRAINING DATA           │
                        │  orders · order_products · products│
                        └────────────┬────────────────────┘
                                     │ temporal split
                          ┌──────────┼──────────┐
                          ▼          ▼          ▼
                        train       val        test
                          │
              ┌───────────┼───────────┐
              │                       │
     ┌────────▼────────┐   ┌──────────▼──────────┐
     │   USER TOWER    │   │    ITEM TOWER        │
     │                 │   │                      │
     │  user_id        │   │  item_id             │
     │     │           │   │  aisle_id ──┐        │
     │  Embedding(128) │   │  dept_id  ──┤        │
     │     │           │   │             ▼        │
     │  Linear(128→256)│   │  Embedding concat    │
     │  ReLU           │   │  (128 + 32 + 16=176) │
     │  Dropout(0.1)   │   │  Linear(176→256)     │
     │  Linear(256→128)│   │  ReLU                │
     │     │           │   │  Dropout(0.1)        │
     │  L2-Norm        │   │  Linear(256→128)     │
     └────────┬────────┘   │     │                │
              │            │  L2-Norm             │
              │            └──────────┬───────────┘
              │                       │
              └──────────┬────────────┘
                         │
              ┌──────────▼────────────┐
              │   InfoNCE Loss (NT-Xent)│
              │   - In-batch negatives  │
              │   - Hard negatives      │
              │   - Popularity debiasing│
              └──────────┬────────────┘
                         │
              ┌──────────▼────────────┐
              │    Trained Model       │
              └──────────┬────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼                               ▼
 ┌───────────────┐             ┌─────────────────┐
 │ All Item Embs │             │  User Embedding  │
 │ (num_items,128)│            │  (query vector)  │
 └───────┬───────┘             └────────┬────────┘
         │                              │
         ▼                              │
 ┌───────────────┐                      │
 │  FAISS Index  │◄─────────────────────┘
 │ IndexFlatIP   │   ANN search, k=200
 └───────┬───────┘
         │  top-200 candidates + scores
         ▼
 ┌────────────────────────────────────┐
 │           RERANKER (GBM)           │
 │  Features per (user, item) pair:   │
 │  · similarity score                │
 │  · log(popularity)                 │
 │  · history flag                    │
 │  · item reorder rate               │
 │  · log(user order count)           │
 │  · log(user history size)          │
 │  CalibratedClassifierCV (sigmoid)  │
 └───────────────┬────────────────────┘
                 │  top-10 / top-20 final recs
                 ▼
         Recall@K · NDCG@K · MRR@K
```

### Two-Stage Pipeline

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| **Retrieval** | Two-tower neural net | User ID | Top-200 candidates (ANN via FAISS) |
| **Reranking** | Gradient Boosting + calibration | 6 hand-crafted features per candidate | Final ranked top-10 / top-20 |

---

## Performance Improvements

### Baseline → Current

| Experiment | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|------------|-----------|---------|-----------|---------|
| Popularity baseline | 0.0699 | 0.0976 | 0.0955 | 0.0974 |
| Two-Tower v1 (embedding lookup only) | 0.1065 | 0.1047 | 0.1427 | 0.1140 |
| Reranked v1 (3 features, 20 candidates) | 0.1369 | 0.1423 | 0.1427 | 0.1326 |
| Two-Tower v2 (MLP + content embeddings) | 0.1451 | 0.1742 | 0.2073 | 0.1854 |
| **Reranked v2 (6 features, 200 candidates)** | **0.1819** | **0.2315** | **0.2073** | **0.2179** |

### What drove each gain

#### Model architecture
| Change | Why it helps |
|--------|-------------|
| MLP user tower (`128→256→128`) | Non-linear projection; model can learn feature interactions instead of just memorizing user IDs |
| MLP item tower (`176→256→128`) | Same, plus fuses content signals into a single embedding |
| Aisle embeddings (dim=32) | Products in the same aisle share structural similarity; aisle signal generalizes across the 50K item catalog |
| Department embeddings (dim=16) | Coarser catalog signal; helps cold-start items with few interactions |
| Kaiming init on linear layers | He initialization is correct for ReLU activations; avoids vanishing/exploding gradients at startup |

#### Training quality
| Change | Why it helps |
|--------|-------------|
| Hard negative mining (per-epoch, 25% user sample) | Forces the model to distinguish near-miss items, not just obviously unrelated ones. Larger gradient signal → faster convergence and better separation in embedding space |
| Popularity debiasing (sampling-bias correction) | Frequent items appear as in-batch negatives far more often than their true relevance warrants. Subtracting `log(q_j)` from off-diagonal logits corrects this bias so the model doesn't over-penalise popular items |
| Temporal validation split | Replacing random `train_test_split` with a temporal holdout (last order per user) prevents future data from leaking into validation, giving an accurate early-stopping signal |
| Patience 2 → 4 | With a harder training objective (hard negs + debiasing), loss curves are noisier. More patience avoids stopping before the model has converged |
| Temperature `0.07` consistent | Validation was previously computed with a hardcoded temperature that could diverge from training; now the same parameter is used everywhere |

#### Retrieval pipeline
| Change | Why it helps |
|--------|-------------|
| Candidate pool 20 → 200 | The reranker can only recover items the retrieval stage found. A wider pool dramatically raises the ceiling on Recall@20 without changing the neural model |
| FAISS fallback removed | The sklearn fallback used `.kneighbors()` but `retrieve_topk` called `.search()` — it would have crashed silently. Surfacing FAISS errors immediately is safer |
| `emb_dim` loaded from `mappings.pt` | Previously hardcoded to 128 in inference; now any trained dimension is loaded automatically |

#### Reranker
| Change | Why it helps |
|--------|-------------|
| 3 features → 6 features | Added `item_reorder_rate`, `log_user_order_count`, `log_user_hist_size`. Reorder rate is a strong purchase-intent signal in grocery; user activity level helps calibrate scores for heavy vs. light users |
| Train on val users, evaluate on test | Original code trained the reranker on test users' ground truth (label leakage), inflating reported metrics. Now the reranker is trained on validation ground truth and evaluated on unseen test users |

---

## File Structure

```
instacart_recsys/
├── src/
│   ├── __init__.py
│   ├── model.py                      ← TwoTowerModel (MLP towers + content embeddings)
│   ├── data_processing.py            ← loading, splitting, content mappings, feature helpers
│   ├── train.py                      ← end-to-end pipeline
│   ├── inference.py                  ← load saved model, run retrieval, evaluate
│   └── evaluate.py                   ← Recall@K, NDCG@K, MRR@K
├── tests/
│   └── test_data_preprocessing.py
├── data/
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   └── products.csv                  ← required for aisle/dept content features
└── models/
    └── version_YYYYMMDD_HHMMSS/
        ├── model.pt                  ← best checkpoint (early stopping)
        ├── mappings.pt               ← user2idx, prod2idx, content tensors, model config
        └── metadata.json             ← hyperparams + eval results
```

---

## How to Run

**Train:**
```bash
python -m src.train
```

Key parameters (all have defaults):
```
--data_dir      path to data/          default: ./data/
--emb_dim       embedding dimension    default: 128
--hidden_dim    MLP hidden size        default: 256
--batch_size    training batch size    default: 4096
--epochs        max epochs             default: 8
--k_retrieve    FAISS candidate pool   default: 200
--num_workers   DataLoader workers     default: 2
```

**Inference on saved model:**
```bash
python -m src.inference --model_dir ./models/version_YYYYMMDD_HHMMSS
```

**Experiment tracking:**
```bash
mlflow ui   # then open http://localhost:5000
```

---

## Key Design Decisions

**Why temporal splits everywhere?**
Train/val/test are all split by `order_number` (last order = test, second-to-last = val, rest = train). Random splits would let future orders leak into training, producing optimistic metrics that don't reflect real next-basket prediction.

**Why in-batch negatives with debiasing?**
In-batch negatives are efficient (no extra forward passes) but biased — popular items appear as negatives far more often than their frequency in the recommendation space justifies. The sampling-bias correction (`logit -= log(q_j)`) from Google's 2019 paper removes this distortion.

**Why a separate reranker instead of end-to-end?**
The two-tower model optimizes for broad recall across millions of users at millisecond latency (via FAISS ANN). The GBM reranker operates on a small candidate set (200 items) and can use features that are expensive to compute at retrieval scale (per-item reorder rates, user behavioral stats). Two-stage is the industry standard for this reason.

**Why save model config in `mappings.pt`?**
`emb_dim`, `hidden_dim`, `num_aisles`, `num_depts` are written into `mappings.pt` at train time and read back at inference time. This means inference never hardcodes architecture parameters — any training configuration loads correctly.
