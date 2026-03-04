import os
import sys
import hashlib
import json
import datetime
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ── Ensure src/ is on sys.path for multiprocessing spawn compatibility ────
# On macOS (Python 3.8+), multiprocessing uses "spawn" which re-imports this
# module in worker processes. Without this, sibling imports like data_processing
# fail with ModuleNotFoundError.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import mlflow
# sklearn imported lazily inside train_reranker() — it is only needed when use_scaler=True
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Absolute imports so the module can be run directly as `python train.py`
# or imported as part of a package — both work without modification.
from data_processing import (
    load_and_merge_data,
    load_products,
    filter_active_users,
    temporal_train_test_split,
    temporal_val_split,
    build_mappings,
    build_content_mappings,
    get_item_content_tensors,
    interactions_to_indices,
    get_popularity,
    get_item_reorder_rates,
    get_user_reorder_rates,
    get_user_stats,
)
from model import TwoTowerModel
from evaluate import evaluate_recommendations, evaluate_orderable_recommendations


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InteractionDataset(Dataset):
    """Yields (user_idx, pos_item_idx) pairs. In-batch negatives are used during training."""

    def __init__(self, interactions_df):
        self.data = (
            interactions_df[['user_idx', 'product_idx']]
            .drop_duplicates()
            .values.astype(np.int64)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i = self.data[idx]
        return int(u), int(i)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(item_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(item_embeddings.shape[1])
    index.add(item_embeddings)
    return index


def retrieve_topk(index, user_embeddings: np.ndarray, k: int = 20):
    scores, indices = index.search(user_embeddings.astype('float32'), k)
    return indices, scores


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    model: TwoTowerModel,
    train_idx,
    item_aisle: torch.LongTensor,
    item_dept: torch.LongTensor,
    device,
    k: int = 50,
    num_hard: int = 5,
    sample_frac: float = 0.15,
) -> dict:
    """
    After each epoch: retrieve top-k items per user, exclude known positives,
    and return the remaining as hard negatives for the next epoch.

    Only mines for a random `sample_frac` fraction of training users each epoch.
    Users not sampled fall back to in-batch negatives — still meaningful signal.
    At 0.25, mining costs ~4× less while negatives stay fresh every epoch.

    Returns dict: user_idx -> list[item_idx]
    """
    model.eval()

    # Build item index from current embeddings (already L2-normalized by model)
    with torch.no_grad():
        item_embs = model.get_all_item_embeddings(item_aisle, item_dept).numpy().astype('float32')
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)

    all_users = train_idx['user_idx'].unique()
    n_sample = max(1, int(len(all_users) * sample_frac))
    users = np.random.choice(all_users, size=n_sample, replace=False)
    user_positives = train_idx.groupby('user_idx')['product_idx'].apply(set).to_dict()

    hard_neg_dict = {}
    batch_sz = 2048
    for i in tqdm(range(0, len(users), batch_sz), desc='Mining hard negs', leave=False):
        batch_users = users[i: i + batch_sz]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
        with torch.no_grad():
            user_emb = model.get_user_embedding(user_tensor).cpu().numpy().astype('float32')
        _, indices = index.search(user_emb, k + num_hard)

        for j, u in enumerate(batch_users):
            pos = user_positives.get(u, set())
            hard = [int(idx) for idx in indices[j] if idx not in pos][:num_hard]
            hard_neg_dict[u] = hard

    model.train()
    return hard_neg_dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_pipeline(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer,
    device,
    item_aisle: torch.LongTensor,
    item_dept: torch.LongTensor,
    item_probs: torch.FloatTensor,
    hard_neg_dict: dict = None,
    user_pos_dict: dict = None,
    temperature: float = 0.07,
    num_hard: int = 5,
    num_semihard: int = 5,
    num_random: int = 5,
    alpha_hard_neg: float = 0.3,
) -> float:
    """
    One epoch of training with:
    - In-batch negatives (main loss)
    - Sampling-bias correction (popularity debiasing) on in-batch negatives
    - Per-user auxiliary loss with negatives from:
      * hard negatives (hard_neg_dict)
      * semi-hard negatives (same department as positive item)
      * random negatives (from item_probs)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    # Build department -> item indices once per epoch on device
    unique_depts = torch.unique(item_dept)
    dept_to_items = {
        int(d.item()): torch.nonzero(item_dept == d, as_tuple=False).squeeze(1)
        for d in unique_depts
    }

    for batch in tqdm(dataloader, desc='train', leave=False):
        user_idx, pos_idx = batch
        user_idx = user_idx.to(device)
        pos_idx = pos_idx.to(device)
        B = user_idx.size(0)

        user_emb = model.get_user_embedding(user_idx)  # (B, D)
        pos_emb = model.get_item_embedding(
            pos_idx, item_aisle[pos_idx], item_dept[pos_idx]
        )  # (B, D)

        # --- In-batch logits (unchanged retrieval objective) ---
        logits_inbatch = (user_emb @ pos_emb.t()) / temperature  # (B, B)

        # --- Popularity debiasing: sampling-bias correction on in-batch negatives ---
        # Subtract log(q_j) from off-diagonal logits for the in-batch portion.
        # Hard negatives are intentionally mined, so no correction is applied to them.
        log_probs = torch.log(item_probs[pos_idx] + 1e-8)  # (B,)
        correction = log_probs.unsqueeze(0).expand(B, B)   # (B, B)
        diag_mask = torch.eye(B, device=device, dtype=torch.bool)
        correction = correction.masked_fill(diag_mask, 0.0)
        logits_inbatch = logits_inbatch - correction

        # --- In-batch loss: diagonal positives ---
        labels = torch.arange(B, device=device)
        loss_inbatch = criterion(logits_inbatch, labels)

        # --- Per-user hard-negative loss ---
        loss_hard = torch.tensor(0.0, device=device)
        if hard_neg_dict is not None:
            batch_users = user_idx.detach().cpu().tolist()

            # 1) hard negatives per user from mining dict
            hard_lists = [hard_neg_dict.get(int(u), [])[:num_hard] for u in batch_users]

            # 2) semi-hard negatives: sample from same department as positive item
            semi_idx = torch.full((B, num_semihard), -1, dtype=torch.long, device=device)
            if num_semihard > 0:
                pos_dept = item_dept[pos_idx]  # (B,)
                for d in torch.unique(pos_dept):
                    rows = torch.nonzero(pos_dept == d, as_tuple=False).squeeze(1)
                    pool = dept_to_items.get(int(d.item()), None)
                    if pool is None or pool.numel() == 0:
                        continue
                    rnd = torch.randint(0, pool.numel(), (rows.numel(), num_semihard), device=device)
                    semi_idx[rows] = pool[rnd]

            # 3) random negatives from global item distribution
            rand_idx = torch.full((B, num_random), -1, dtype=torch.long, device=device)
            if num_random > 0:
                rand_idx = torch.multinomial(item_probs, num_samples=B * num_random, replacement=True).view(B, num_random)

            # Merge & filter per-user (exclude train positives and current positive)
            neg_lists = []
            for i, u in enumerate(batch_users):
                pos_set = user_pos_dict.get(int(u), set()) if user_pos_dict is not None else set()
                cur_pos = int(pos_idx[i].item())

                merged = []
                merged.extend(hard_lists[i])
                if num_semihard > 0:
                    merged.extend([int(x) for x in semi_idx[i].tolist() if x >= 0])
                if num_random > 0:
                    merged.extend([int(x) for x in rand_idx[i].tolist() if x >= 0])

                # stable dedupe while filtering positives
                seen = set()
                filtered = []
                for it in merged:
                    if it == cur_pos or it in pos_set or it in seen:
                        continue
                    seen.add(it)
                    filtered.append(it)
                neg_lists.append(filtered)

            Hmax = max((len(h) for h in neg_lists), default=0)

            if Hmax > 0:
                hard_idx = torch.full((B, Hmax), -1, dtype=torch.long, device=device)
                hard_mask = torch.zeros((B, Hmax), dtype=torch.bool, device=device)

                for i, h in enumerate(neg_lists):
                    if not h:
                        continue
                    h_tensor = torch.tensor(h, dtype=torch.long, device=device)
                    h_len = h_tensor.numel()
                    hard_idx[i, :h_len] = h_tensor
                    hard_mask[i, :h_len] = True

                # Safe embedding lookup; padded entries are masked out afterwards.
                safe_hard_idx = torch.clamp(hard_idx, min=0)
                hard_emb = model.get_item_embedding(
                    safe_hard_idx.view(-1),
                    item_aisle[safe_hard_idx.view(-1)],
                    item_dept[safe_hard_idx.view(-1)],
                ).view(B, Hmax, -1)  # (B, Hmax, D)

                hard_logits = (hard_emb * user_emb.unsqueeze(1)).sum(dim=2) / temperature  # (B, Hmax)
                hard_logits = hard_logits.masked_fill(~hard_mask, -1e9)

                pos_logits = (user_emb * pos_emb).sum(dim=1, keepdim=True) / temperature  # (B, 1)
                logits_hard = torch.cat([pos_logits, hard_logits], dim=1)  # (B, 1+Hmax)

                valid_rows = hard_mask.any(dim=1)
                if valid_rows.any():
                    hard_labels = torch.zeros(valid_rows.sum(), dtype=torch.long, device=device)
                    loss_hard = criterion(logits_hard[valid_rows], hard_labels)

        loss = loss_inbatch + alpha_hard_neg * loss_hard

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B

    return total_loss / len(dataloader.dataset)


def compute_val_loss(
    model: TwoTowerModel,
    val_dataloader: DataLoader,
    device,
    item_aisle: torch.LongTensor,
    item_dept: torch.LongTensor,
    temperature: float = 0.07,
) -> float:
    """Validation loss using the same InfoNCE objective (no hard negatives, no debiasing)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='val', leave=False):
            user_idx, pos_idx = batch
            user_idx = user_idx.to(device)
            pos_idx = pos_idx.to(device)
            B = user_idx.size(0)

            user_emb = model.get_user_embedding(user_idx)
            pos_emb = model.get_item_embedding(
                pos_idx, item_aisle[pos_idx], item_dept[pos_idx]
            )
            logits = (user_emb @ pos_emb.t()) / temperature
            labels = torch.arange(B, device=device)
            total_loss += criterion(logits, labels).item() * B

    return total_loss / len(val_dataloader.dataset)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def extract_features_with_truth(
    candidates: np.ndarray,
    cand_scores: np.ndarray,
    users: list,
    user_to_row: dict,
    user_hist: dict,
    truths: dict,
    pop_counts: np.ndarray,
    item_reorder_rates: np.ndarray,
    user_order_counts: np.ndarray,
    user_hist_sizes: np.ndarray,
):
    """
    Extract 6-dimensional features for reranking:
      [sim_score, log_pop, history_flag, item_reorder_rate,
       log_user_order_count, log_user_hist_size]

    Args:
        candidates: (num_users, k) retrieved item indices
        cand_scores: (num_users, k) similarity scores
        users: user_idx list to process
        user_to_row: user_idx → row index in candidates/cand_scores
        user_hist: user_idx → set of training item indices
        truths: user_idx → list of held-out item indices (ground truth)
        pop_counts: (num_items,) global purchase counts
        item_reorder_rates: (num_items,) fraction of reorders per item
        user_order_counts: (num_users,) total orders per user
        user_hist_sizes: (num_users,) distinct items in training per user

    Returns:
        features: (N, 6) float32 array
        labels: (N,) int array  (1 = item in ground truth)
    """
    features, labels = [], []

    for u in tqdm(users, desc='Extracting features', leave=False):
        row_i = user_to_row[u]
        items = candidates[row_i]
        sims = cand_scores[row_i]
        hist = user_hist.get(u, set())
        truth_set = set(truths.get(u, []))

        log_pop = np.log1p(pop_counts[items]).astype(np.float32)
        reorder_rate = item_reorder_rates[items]
        log_uoc = float(np.log1p(user_order_counts[u]))
        log_uhs = float(np.log1p(user_hist_sizes[u]))

        for i, item in enumerate(items):
            features.append([
                float(sims[i]),
                float(log_pop[i]),
                1.0 if item in hist else 0.0,
                float(reorder_rate[i]),
                log_uoc,
                log_uhs,
            ])
            labels.append(1 if item in truth_set else 0)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def train_reranker(
    features: np.ndarray,
    labels: np.ndarray,
    group: List[int],
    use_scaler: bool = False,
):
    """Train a LightGBM LambdaRank reranker with per-user query groups."""
    try:
        from lightgbm import LGBMRanker
    except ImportError as e:
        raise ImportError(
            "lightgbm is required for reranker training. Install it with: pip install lightgbm"
        ) from e

    if sum(group) != len(features):
        raise ValueError(f"Invalid group sizes: sum(group)={sum(group)} != n_samples={len(features)}")

    scaler = None
    train_x = features
    if use_scaler:
        from sklearn.preprocessing import StandardScaler  # lazy import — only needed when use_scaler=True
        print(f'  Scaling features ({len(features)} samples, {features.shape[1]} features)...')
        scaler = StandardScaler()
        train_x = scaler.fit_transform(features)

    print('  Training LGBMRanker (objective=lambdarank, metric=ndcg)...')
    ranker = LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1,
    )
    ranker.fit(train_x, labels, group=group)
    print('  Reranker training complete.')
    return scaler, ranker


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_metadata(version, model_dir, emb_dim, hidden_dim, batch_size, epochs, results, rerank_results):
    metadata = {
        'version': version,
        'model_dir': model_dir,
        'emb_dim': emb_dim,
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'epochs': epochs,
        'results': results,
        'rerank_results': rerank_results,
    }
    path = os.path.join(model_dir, 'metadata.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f'Metadata saved to {path}')


def is_orderable(item_idx: int, user_idx: int, seed: int = 42, keep_prob: float = 0.85) -> bool:
    """Deterministic mock orderability by (user, item), simulating fulfillment constraints."""
    key = f"{seed}|{user_idx}|{item_idx}".encode('utf-8')
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], byteorder='big', signed=False) / float(2**64)
    return value < keep_prob


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_dir: str = './data/',
    emb_dim: int = 128,
    hidden_dim: int = 256,
    batch_size: int = 4096,
    epochs: int = 8,
    temperature: float = 0.07,
    num_workers: int = 4,
    k_retrieve: int = 200,
    seed: int = 42,
    num_hard: int = 5,
    num_semihard: int = 5,
    num_random: int = 5,
    alpha_hard_neg: float = 0.5,
):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'./models/version_{version}'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pt')
    mappings_path = os.path.join(model_dir, 'mappings.pt')

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    print('Loading and preprocessing data...')
    orders, interactions, _ = load_and_merge_data(data_dir)
    products = load_products(data_dir)
    interactions = filter_active_users(orders, interactions, min_orders=3)
    train_df, test_df = temporal_train_test_split(interactions)

    # Temporal validation split (no random shuffling)
    train_df, val_df = temporal_val_split(train_df)

    user2idx, prod2idx = build_mappings(train_df)
    aisle2idx, dept2idx = build_content_mappings(products)
    item_aisle, item_dept = get_item_content_tensors(prod2idx, aisle2idx, dept2idx, products)

    train_idx, train_drop_stats = interactions_to_indices(train_df, user2idx, prod2idx)
    val_idx, val_drop_stats = interactions_to_indices(val_df, user2idx, prod2idx)
    test_idx, test_drop_stats = interactions_to_indices(test_df, user2idx, prod2idx)
    train_user_pos = train_idx.groupby('user_idx')['product_idx'].apply(lambda s: set(s.tolist())).to_dict()

    print(
        f"OOV drops (val): rows={val_drop_stats['dropped_rows_count']} "
        f"unique_products={val_drop_stats['dropped_unique_products_count']} "
        f"fraction={val_drop_stats['dropped_fraction']:.6f}"
    )
    print(
        f"OOV drops (test): rows={test_drop_stats['dropped_rows_count']} "
        f"unique_products={test_drop_stats['dropped_unique_products_count']} "
        f"fraction={test_drop_stats['dropped_fraction']:.6f}"
    )

    num_users = len(user2idx)
    num_items = len(prod2idx)
    num_aisles = len(aisle2idx)
    num_depts = len(dept2idx)
    print(f'Users: {num_users}, Items: {num_items}, Aisles: {num_aisles}, Depts: {num_depts}')

    # Item sampling probabilities for popularity debiasing (on device)
    counts = train_idx['product_idx'].value_counts()
    item_freq = np.ones(num_items, dtype=np.float32)  # Laplace smoothing
    item_freq[counts.index.values] += counts.values.astype(np.float32)
    item_probs = torch.tensor(item_freq / item_freq.sum(), dtype=torch.float32)

    # Reranker feature arrays (computed from training data)
    item_reorder_rates = get_item_reorder_rates(train_idx, num_items)
    user_order_counts, user_hist_sizes = get_user_stats(train_idx, orders, user2idx)

    # Popularity counts array for reranker
    pop_series = train_idx['product_idx'].value_counts()
    pop_counts = np.zeros(num_items, dtype=np.int64)
    pop_counts[pop_series.index.values] = pop_series.values

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    dataset = InteractionDataset(train_idx)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataset = InteractionDataset(val_idx)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_aisles=num_aisles,
        num_depts=num_depts,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
    )
    model.to(device)

    # Move content tensors and item_probs to device once (reused every batch)
    item_aisle = item_aisle.to(device)
    item_dept = item_dept.to(device)
    item_probs = item_probs.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    patience = 4  # increased from 2 — avoids premature early stopping
    patience_counter = 0
    hard_neg_dict = None  # populated after epoch 1

    mlflow.set_experiment('Two-Tower Recsys')
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params({
            'embedding_dim': emb_dim,
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'epochs': epochs,
            'temperature': temperature,
            'patience': patience,
            'num_hard': num_hard,
            'num_semihard': num_semihard,
            'num_random': num_random,
            'alpha_hard_neg': alpha_hard_neg,
        })
        mlflow.log_metric('oov_val_dropped_fraction', val_drop_stats['dropped_fraction'])
        mlflow.log_metric('oov_test_dropped_fraction', test_drop_stats['dropped_fraction'])

        epoch_bar = tqdm(range(1, epochs + 1), desc='Epochs', unit='epoch')
        for epoch in epoch_bar:
            train_loss = train_pipeline(
                model, dataloader, optimizer, device,
                item_aisle=item_aisle,
                item_dept=item_dept,
                item_probs=item_probs,
                hard_neg_dict=hard_neg_dict,
                user_pos_dict=train_user_pos,
                temperature=temperature,
                num_hard=num_hard,
                num_semihard=num_semihard,
                num_random=num_random,
                alpha_hard_neg=alpha_hard_neg,
            )

            val_loss = compute_val_loss(
                model, val_dataloader, device,
                item_aisle=item_aisle,
                item_dept=item_dept,
                temperature=temperature,
            )

            epoch_bar.set_postfix(train=f'{train_loss:.4f}', val=f'{val_loss:.4f}')
            tqdm.write(f'Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)

            # Early stopping: save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write('Early stopping triggered.')
                    break

            # Mine hard negatives every 2 epochs (expensive FAISS search)
            if epoch % 2 == 0:
                tqdm.write('  Mining hard negatives...')
                hard_neg_dict = mine_hard_negatives(
                    model, train_idx, item_aisle, item_dept, device, k=50, num_hard=5
                )

    # Reload best checkpoint
    # weights_only=True: model.pt contains only tensor weights (state_dict) — safe to restrict.
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Save mappings (includes content info for inference)
    torch.save(
        {
            'user2idx': user2idx,
            'prod2idx': prod2idx,
            'aisle2idx': aisle2idx,
            'dept2idx': dept2idx,
            'item_aisle': item_aisle.cpu(),
            'item_dept': item_dept.cpu(),
            'model_config': {
                'num_aisles': num_aisles,
                'num_depts': num_depts,
                'emb_dim': emb_dim,
                'hidden_dim': hidden_dim,
            },
        },
        mappings_path,
    )
    print(f'Model saved to {model_path}')

    # ------------------------------------------------------------------
    # Build FAISS index for retrieval
    # ------------------------------------------------------------------
    print('Building FAISS index...')
    with torch.no_grad():
        item_embs = model.get_all_item_embeddings(item_aisle, item_dept).numpy().astype('float32')
    # Model outputs are already L2-normalized
    index = build_faiss_index(item_embs)
    print(f'Index built. Item embedding shape: {item_embs.shape}')

    # ------------------------------------------------------------------
    # Evaluate retrieval model on test users
    # ------------------------------------------------------------------
    test_by_user = test_idx.groupby('user_idx')['product_idx'].apply(list).to_dict()
    test_users = list(test_by_user.keys())

    with torch.no_grad():
        test_user_tensor = torch.tensor(test_users, dtype=torch.long, device=device)
        test_user_embs = model.get_user_embedding(test_user_tensor).cpu().numpy().astype('float32')
    # No faiss.normalize_L2 needed — get_user_embedding() calls F.normalize internally.

    K_list = [10, 20]
    # Retrieve a wide candidate pool — reranker selects from all k_retrieve items.
    # Retrieval baseline is evaluated on top max(K_list) only (FAISS order).
    recs_idx, recs_scores = retrieve_topk(index, test_user_embs, k=k_retrieve)
    test_user_to_row = {u: i for i, u in enumerate(test_users)}
    recs = {u: recs_idx[i][:max(K_list)].tolist() for i, u in enumerate(test_users)}
    truths = test_by_user

    results = evaluate_recommendations(recs, truths, ks=K_list)
    retrieval_orderable_results = evaluate_orderable_recommendations(
        recs,
        truths,
        is_orderable=lambda item_idx, user_idx: is_orderable(item_idx, user_idx, seed=seed),
        ks=K_list,
    )

    # Popularity baseline
    pop_series_sorted = train_idx['product_idx'].value_counts()
    popular_items = pop_series_sorted.index.tolist()
    pop_recs = {u: popular_items[:max(K_list)] for u in test_users}
    pop_results = evaluate_recommendations(pop_recs, truths, ks=K_list)

    # ------------------------------------------------------------------
    # Reranker: train on val users, evaluate on test users
    # ------------------------------------------------------------------
    print('\nTraining reranker on validation users (no leakage from test)...')

    val_by_user = val_idx.groupby('user_idx')['product_idx'].apply(list).to_dict()
    valid_user_set = set(user2idx.values())
    val_users = [u for u in val_by_user.keys() if u in valid_user_set]

    # Retrieve candidates for val users
    with torch.no_grad():
        val_user_tensor = torch.tensor(val_users, dtype=torch.long, device=device)
        val_user_embs = model.get_user_embedding(val_user_tensor).cpu().numpy().astype('float32')
    # No faiss.normalize_L2 needed — get_user_embedding() calls F.normalize internally.
    val_recs_idx, val_recs_scores = retrieve_topk(index, val_user_embs, k=k_retrieve)
    val_user_to_row = {u: i for i, u in enumerate(val_users)}

    # Build user history from training data (excludes val/test)
    train_user_hist = train_idx.groupby('user_idx')['product_idx'].apply(lambda s: set(s.tolist())).to_dict()

    # Extract features for val users to train reranker
    train_feat, train_labels = extract_features_with_truth(
        candidates=val_recs_idx,
        cand_scores=val_recs_scores,
        users=val_users,
        user_to_row=val_user_to_row,
        user_hist=train_user_hist,
        truths=val_by_user,
        pop_counts=pop_counts,
        item_reorder_rates=item_reorder_rates,
        user_order_counts=user_order_counts,
        user_hist_sizes=user_hist_sizes,
    )

    pos_rate = train_labels.mean()
    print(f'  Positive label rate in reranker training: {pos_rate:.4f} ({len(train_feat)} samples)')

    train_group = [len(val_recs_idx[val_user_to_row[u]]) for u in val_users]
    if any(g != k_retrieve for g in train_group):
        print('  Warning: some reranker groups differ from k_retrieve.')

    scaler, reranker = train_reranker(
        train_feat,
        train_labels,
        group=train_group,
        use_scaler=False,
    )

    # Rerank test users (no leakage: reranker never saw test ground truth)
    reranked_recs = {}
    for u in tqdm(test_users, desc='Reranking'):
        row_i = test_user_to_row[u]
        items = recs_idx[row_i]
        sims = recs_scores[row_i]
        hist = train_user_hist.get(u, set())

        import pandas as pd
        feature_names = [
            "sim_score",
            "log_pop",
            "history_flag",
            "item_reorder_rate",
            "log_user_order_count",
            "log_user_hist_size",
        ]
        item_features = np.array([
            [
                float(sims[i]),
                float(np.log1p(pop_counts[items[i]])),
                1.0 if items[i] in hist else 0.0,
                float(item_reorder_rates[items[i]]),
                float(np.log1p(user_order_counts[u])),
                float(np.log1p(user_hist_sizes[u])),
            ]
            for i in range(len(items))
        ], dtype=np.float32)

        if scaler is not None:
            item_features = scaler.transform(item_features)
        item_features_df = pd.DataFrame(item_features, columns=feature_names)
        scores = reranker.predict(item_features_df)
        order = np.argsort(-scores)
        reranked_recs[u] = items[order].tolist()

    rerank_results = evaluate_recommendations(reranked_recs, truths, ks=K_list)
    rerank_orderable_results = evaluate_orderable_recommendations(
        reranked_recs,
        truths,
        is_orderable=lambda item_idx, user_idx: is_orderable(item_idx, user_idx, seed=seed),
        ks=K_list,
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print('\nRetrieval Results (Two-Tower — all test users):')
    for k in K_list:
        print(f"  Recall@{k}: {results[f'Recall@{k}']:.4f}  NDCG@{k}: {results[f'NDCG@{k}']:.4f}  MRR@{k}: {results[f'MRR@{k}']:.4f}")

    print('\nInventory-Constrained Retrieval Results (Two-Tower — all test users):')
    for k in K_list:
        print(
            f"  OrderableRecall@{k}: {retrieval_orderable_results[f'OrderableRecall@{k}']:.4f}  "
            f"OrderableRate@{k}: {retrieval_orderable_results[f'OrderableRate@{k}']:.4f}"
        )

    print('\nPopularity Baseline (all test users):')
    for k in K_list:
        print(f"  Recall@{k}: {pop_results[f'Recall@{k}']:.4f}  NDCG@{k}: {pop_results[f'NDCG@{k}']:.4f}  MRR@{k}: {pop_results[f'MRR@{k}']:.4f}")

    print(f'\nReranked Results (all test users, reranker trained on val):')
    for k in K_list:
        print(f"  Recall@{k}: {rerank_results[f'Recall@{k}']:.4f}  NDCG@{k}: {rerank_results[f'NDCG@{k}']:.4f}  MRR@{k}: {rerank_results[f'MRR@{k}']:.4f}")

    print('\nInventory-Constrained Reranked Results (all test users):')
    for k in K_list:
        print(
            f"  OrderableRecall@{k}: {rerank_orderable_results[f'OrderableRecall@{k}']:.4f}  "
            f"OrderableRate@{k}: {rerank_orderable_results[f'OrderableRate@{k}']:.4f}"
        )

    with mlflow.start_run(run_id=run_id):
        def sanitize_metric_name(name):
            return name.replace("@", "_")
        for k in K_list:
            mlflow.log_metric(sanitize_metric_name(f"retrieval_orderable_recall@{k}"), retrieval_orderable_results[f"OrderableRecall@{k}"])
            mlflow.log_metric(sanitize_metric_name(f"retrieval_orderable_rate@{k}"), retrieval_orderable_results[f"OrderableRate@{k}"])
            mlflow.log_metric(sanitize_metric_name(f"rerank_orderable_recall@{k}"), rerank_orderable_results[f"OrderableRecall@{k}"])
            mlflow.log_metric(sanitize_metric_name(f"rerank_orderable_rate@{k}"), rerank_orderable_results[f"OrderableRate@{k}"])

    save_metadata(
        version=version,
        model_dir=model_dir,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        epochs=epochs,
        results=results,
        rerank_results=rerank_results,
    )


if __name__ == '__main__':
    main()