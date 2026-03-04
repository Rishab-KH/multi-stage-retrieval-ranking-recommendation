import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple


def load_and_merge_data(data_dir: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load orders and order_products (prior + train) and merge into a single interaction frame.
    Interactions include the 'reordered' flag for downstream feature computation.
    """
    orders_path = os.path.join(data_dir, 'orders.csv')
    prior_path = os.path.join(data_dir, 'order_products__prior.csv')
    train_path = os.path.join(data_dir, 'order_products__train.csv')

    orders = pd.read_csv(orders_path)
    prior = pd.read_csv(prior_path)
    train = pd.read_csv(train_path)

    order_products = pd.concat([prior, train], ignore_index=True)

    interactions = orders.merge(order_products, on='order_id', how='inner')
    interactions = interactions[['user_id', 'order_id', 'order_number', 'product_id', 'reordered']]
    return orders, interactions, order_products


def load_products(data_dir: str = '.') -> pd.DataFrame:
    """Load product catalog: product_id, aisle_id, department_id."""
    return pd.read_csv(
        os.path.join(data_dir, 'products.csv'),
        usecols=['product_id', 'aisle_id', 'department_id'],
    )


def filter_active_users(orders: pd.DataFrame, interactions: pd.DataFrame, min_orders: int = 3):
    """
    Keep only users with at least `min_orders` distinct orders.

    Uses nunique(order_id) rather than max(order_number) to count orders. max(order_number)
    only works correctly when order_number is a dense 1-based sequential counter per user
    (true for Instacart, but fragile in general). Counting unique order_ids is always correct.
    """
    user_order_counts = orders.groupby('user_id')['order_id'].nunique()
    active_users = user_order_counts[user_order_counts >= min_orders].index
    return interactions[interactions['user_id'].isin(active_users)].copy()


def temporal_train_test_split(interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user: last order → test, everything before → train.
    Temporal split ensures no future information leaks into training.
    """
    last = interactions.groupby('user_id')['order_number'].max().rename('last_order').reset_index()
    df = interactions.merge(last, on='user_id')
    cols = ['user_id', 'order_id', 'order_number', 'product_id', 'reordered']
    train = df[df['order_number'] < df['last_order']][cols].copy()
    test = df[df['order_number'] == df['last_order']][cols].copy()
    return train, test


def temporal_val_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From the training split, hold out each user's last order as validation.
    Mirrors the test split logic — no random shuffling, strictly temporal.

    Note: users with exactly one training order will have that order placed in val,
    leaving their train portion empty. This is expected and handled downstream —
    the min_orders=3 filter on the full dataset ensures all users reaching this
    function have at least 2 training orders (test already consumed the last one),
    so in practice every user retains at least one order in train after this split.
    """
    last = train_df.groupby('user_id')['order_number'].max().rename('last_train_order').reset_index()
    df = train_df.merge(last, on='user_id')
    cols = ['user_id', 'order_id', 'order_number', 'product_id', 'reordered']
    val = df[df['order_number'] == df['last_train_order']][cols].copy()
    train = df[df['order_number'] < df['last_train_order']][cols].copy()
    return train, val


def build_mappings(train: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Compact integer index mappings for users and products.
    Both vocabularies are built from training data only — avoids test leakage.
    Test users/products not seen in training will be treated as OOV and dropped
    by interactions_to_indices(), which reports drop statistics transparently.
    """
    users = pd.Index(train['user_id'].unique()).sort_values()
    user2idx = {u: i for i, u in enumerate(users)}

    products = pd.Index(train['product_id'].unique()).sort_values()
    prod2idx = {p: i for i, p in enumerate(products)}
    return user2idx, prod2idx


def build_content_mappings(products: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Build compact integer mappings for aisle_id and department_id."""
    aisles = pd.Index(products['aisle_id'].unique()).sort_values()
    aisle2idx = {a: i for i, a in enumerate(aisles)}

    depts = pd.Index(products['department_id'].unique()).sort_values()
    dept2idx = {d: i for i, d in enumerate(depts)}
    return aisle2idx, dept2idx


def get_item_content_tensors(
    prod2idx: Dict,
    aisle2idx: Dict,
    dept2idx: Dict,
    products: pd.DataFrame,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Build per-item content tensors:
      item_aisle[item_idx] → aisle compact index
      item_dept[item_idx]  → department compact index
    Items missing from products.csv default to index 0.
    """
    num_items = len(prod2idx)
    item_aisle = torch.zeros(num_items, dtype=torch.long)
    item_dept = torch.zeros(num_items, dtype=torch.long)

    prod_to_aisle = dict(zip(products['product_id'], products['aisle_id']))
    prod_to_dept = dict(zip(products['product_id'], products['department_id']))

    for prod_id, idx in prod2idx.items():
        a = prod_to_aisle.get(prod_id)
        d = prod_to_dept.get(prod_id)
        if a is not None and a in aisle2idx:
            item_aisle[idx] = aisle2idx[a]
        if d is not None and d in dept2idx:
            item_dept[idx] = dept2idx[d]

    return item_aisle, item_dept


def interactions_to_indices(
    df: pd.DataFrame, user2idx: Dict, prod2idx: Dict
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Map raw ids → compact indices. Rows with unknown products are dropped.
    Preserves 'reordered' column when present.

    Returns:
      df_idx: indexed interactions
      dropped_stats: transparency metrics for product OOV drops
        - dropped_rows_count
        - dropped_unique_products_count
        - dropped_fraction
    """
    df = df.copy()

    total_rows = len(df)
    product_idx_mapped = df['product_id'].map(prod2idx)
    oov_mask = product_idx_mapped.isna()
    dropped_rows_count = int(oov_mask.sum())
    dropped_unique_products_count = int(df.loc[oov_mask, 'product_id'].nunique())
    dropped_fraction = float(dropped_rows_count / total_rows) if total_rows > 0 else 0.0

    df['user_idx'] = df['user_id'].map(user2idx)
    df['product_idx'] = product_idx_mapped
    df = df.dropna(subset=['user_idx', 'product_idx'])
    df['user_idx'] = df['user_idx'].astype(int)
    df['product_idx'] = df['product_idx'].astype(int)
    base_cols = ['user_id', 'order_id', 'order_number', 'product_id', 'user_idx', 'product_idx']
    extra = [c for c in ['reordered'] if c in df.columns]
    dropped_stats = {
        'dropped_rows_count': dropped_rows_count,
        'dropped_unique_products_count': dropped_unique_products_count,
        'dropped_fraction': dropped_fraction,
    }
    return df[base_cols + extra], dropped_stats


def get_popularity(train: pd.DataFrame, prod2idx: Dict):
    """Global popularity ranking by training frequency. Returns list of product_idx desc."""
    counts = train['product_id'].map(prod2idx).dropna().astype(int).value_counts()
    return counts.index.tolist()


def get_item_reorder_rates(train_indexed: pd.DataFrame, num_items: int) -> np.ndarray:
    """
    Per-item reorder rate = fraction of purchases that were reorders.
    Items with no data default to 0.0.
    """
    rates = np.zeros(num_items, dtype=np.float32)
    if 'reordered' not in train_indexed.columns:
        return rates
    grp = train_indexed.groupby('product_idx')['reordered'].agg(['sum', 'count'])
    # groupby().agg() never produces zero-count groups, so no need to filter
    rates[grp.index.values] = (grp['sum'] / grp['count']).values.astype(np.float32)
    return rates


def get_user_reorder_rates(train_indexed: pd.DataFrame, num_users: int) -> np.ndarray:
    """Per-user reorder rate = fraction of their training purchases that were reorders."""
    rates = np.zeros(num_users, dtype=np.float32)
    if 'reordered' not in train_indexed.columns:
        return rates
    grp = train_indexed.groupby('user_idx')['reordered'].agg(['sum', 'count'])
    rates[grp.index.values] = (grp['sum'] / grp['count']).values.astype(np.float32)
    return rates


def get_user_stats(
    train_indexed: pd.DataFrame,
    orders: pd.DataFrame,
    user2idx: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        user_order_counts[user_idx]: total order count per user (from orders table)
        user_hist_sizes[user_idx]:   distinct items purchased in training per user
    """
    num_users = len(user2idx)
    user_order_counts = np.zeros(num_users, dtype=np.float32)
    user_hist_sizes = np.zeros(num_users, dtype=np.float32)

    order_cnt = orders.groupby('user_id')['order_number'].max()
    for uid, cnt in order_cnt.items():
        idx = user2idx.get(uid)
        if idx is not None:
            user_order_counts[idx] = cnt

    hist = train_indexed.groupby('user_idx')['product_idx'].nunique()
    user_hist_sizes[hist.index.values] = hist.values.astype(np.float32)

    return user_order_counts, user_hist_sizes