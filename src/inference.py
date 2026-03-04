import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from typing import Optional

import torch
import faiss
import pandas as pd
from pathlib import Path

# Absolute imports — works both as `python inference.py` and when imported as a package module.
from .model import TwoTowerModel
from .data_processing import (
    load_and_merge_data,
    filter_active_users,
    temporal_train_test_split,
    interactions_to_indices,
)
from .evaluate import evaluate_recommendations


def resolve_model_dir(model_dir: Optional[str]) -> str:
    """Resolve model directory; auto-pick latest version if not provided."""
    if model_dir:
        candidate = Path(model_dir)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    models_root = Path('./models')
    version_dirs = sorted(
        [p for p in models_root.glob('version_*') if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not version_dirs:
        raise FileNotFoundError("No model versions found in ./models (expected ./models/version_*)")
    return str(version_dirs[0])


def load_model_and_mappings(model_dir: str):
    """
    Load the trained model and all mappings/content tensors from a model directory.
    Model architecture parameters are read from the saved mappings file — no hardcoding.
    """
    model_path = os.path.join(model_dir, 'model.pt')
    mappings_path = os.path.join(model_dir, 'mappings.pt')

    # weights_only=False is required here because mappings.pt contains Python dicts and
    # tensors serialized together. Only load files you produced yourself from trusted sources.
    mappings = torch.load(mappings_path, map_location='cpu', weights_only=False)
    user2idx = mappings['user2idx']
    prod2idx = mappings['prod2idx']
    item_aisle = mappings['item_aisle']
    item_dept = mappings['item_dept']
    cfg = mappings['model_config']

    model = TwoTowerModel(
        num_users=len(user2idx),
        num_items=len(prod2idx),
        num_aisles=cfg['num_aisles'],
        num_depts=cfg['num_depts'],
        emb_dim=cfg['emb_dim'],
        hidden_dim=cfg['hidden_dim'],
    )
    # weights_only=True is safe here — model.pt contains only tensor weights (state_dict).
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    return model, user2idx, prod2idx, item_aisle, item_dept


def build_faiss_index(model: TwoTowerModel, item_aisle: torch.LongTensor, item_dept: torch.LongTensor):
    """Build a FAISS inner-product index over L2-normalized item embeddings (model outputs are already normalized)."""
    with torch.no_grad():
        item_embs = model.get_all_item_embeddings(item_aisle, item_dept).detach().cpu().numpy().astype('float32')
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)
    return index


def infer_batch(model: TwoTowerModel, user_indices: list, index, k: int = 20):
    """Retrieve top-k items for a batch of users."""
    user_idx_tensor = torch.tensor(user_indices, dtype=torch.long)
    with torch.no_grad():
        user_embs = model.get_user_embedding(user_idx_tensor).detach().cpu().numpy().astype('float32')
    # No faiss.normalize_L2 needed here — get_user_embedding() calls F.normalize internally,
    # so embeddings are already on the unit sphere before FAISS search.
    scores, indices = index.search(user_embs, k)  # FAISS returns (D, I)
    return indices, scores


def main(model_dir=None, data_dir='./data/', k=20, num_users=10):
    """
    Run inference on test users and evaluate recommendations.

    Args:
        model_dir: Path to the saved model directory (default: auto-select latest)
        data_dir: Path to the data directory
        k: Number of recommendations to retrieve
        num_users: Number of users to show detailed results for (-1 for all)
    """
    model_dir = resolve_model_dir(model_dir)
    print(f'Loading model and mappings from: {model_dir}')
    model, user2idx, prod2idx, item_aisle, item_dept = load_model_and_mappings(model_dir)

    idx2prod = {prod_idx: prod_id for prod_id, prod_idx in prod2idx.items()}
    idx2user = {user_idx: user_id for user_id, user_idx in user2idx.items()}

    products_csv = os.path.join(data_dir, 'products.csv')
    prod_id_to_name = {}
    if os.path.exists(products_csv):
        products_df = pd.read_csv(products_csv, usecols=['product_id', 'product_name'])
        prod_id_to_name = dict(zip(products_df['product_id'], products_df['product_name']))
    idx2name = {idx: prod_id_to_name.get(prod_id, str(prod_id)) for prod_id, idx in prod2idx.items()}

    print('Loading and preprocessing data...')
    orders, interactions, _ = load_and_merge_data(data_dir)
    interactions = filter_active_users(orders, interactions, min_orders=3)
    _train_df, test_df = temporal_train_test_split(interactions)

    test_idx, _drop_stats = interactions_to_indices(test_df, user2idx, prod2idx)
    test_by_user = test_idx.groupby('user_idx')['product_idx'].apply(list).to_dict()
    test_users = list(test_by_user.keys())

    print(f'Total test users: {len(test_users)}')

    print('Building FAISS index...')
    index = build_faiss_index(model, item_aisle, item_dept)

    print(f'Running inference for {len(test_users)} users...')
    recs_idx, recs_scores = infer_batch(model, test_users, index, k=k)

    recs = {u: recs_idx[i].tolist() for i, u in enumerate(test_users)}
    scores_dict = {u: recs_scores[i].tolist() for i, u in enumerate(test_users)}

    # Skip users with empty ground truth for metric computation
    metric_users = [u for u in test_users if len(test_by_user.get(u, [])) > 0]
    recs_for_metrics = {u: recs[u] for u in metric_users}
    truths_for_metrics = {u: test_by_user[u] for u in metric_users}

    K_list = [10, 20]
    if len(metric_users) == 0:
        print('\nNo users with non-empty ground truth found. Skipping metric evaluation.')
    else:
        print(f"Evaluating on {len(metric_users)} users with non-empty ground truth (out of {len(test_users)} total).")
        results = evaluate_recommendations(recs_for_metrics, truths_for_metrics, ks=K_list)

        print('\nRetrieval Results (Model):')
        for k in K_list:
            print(
                f"Recall@{k}: {results[f'Recall@{k}']:.4f}, "
                f"NDCG@{k}: {results[f'NDCG@{k}']:.4f}, "
                f"MRR@{k}: {results[f'MRR@{k}']:.4f}"
            )

    users_to_show = test_users[:num_users] if num_users > 0 else test_users

    print('\n' + '=' * 50)
    print(f'Detailed Results for {len(users_to_show)} Users:')
    print('=' * 50)

    for user_idx in users_to_show:
        user_id = idx2user.get(user_idx, user_idx)
        ground_truth_items = test_by_user.get(user_idx, [])
        recommended_items = recs[user_idx][:10]
        recommended_scores = scores_dict[user_idx][:10]

        gt_product_ids = [idx2prod.get(idx, idx) for idx in ground_truth_items]
        hits = set(recommended_items) & set(ground_truth_items)

        print(f'\nUser {user_id} (idx={user_idx}):')
        print(f'  Ground truth items ({len(ground_truth_items)}): {gt_product_ids[:5]}')
        print(f'  Recommended items (top 10):')
        for item_idx, score in zip(recommended_items, recommended_scores):
            prod_id = idx2prod.get(item_idx, item_idx)
            name = idx2name.get(item_idx, str(item_idx))
            hit_marker = ' *HIT*' if item_idx in ground_truth_items else ''
            print(f'    Product {prod_id} {name} (score: {score:.4f}){hit_marker}')
        print(f'  Hits@10: {len(hits)}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on the Two-Tower model')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to the model directory (default: auto-select latest)')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='Path to the data directory')
    parser.add_argument('--k', type=int, default=20,
                        help='Number of recommendations to retrieve')
    parser.add_argument('--num_users', type=int, default=10,
                        help='Number of users to show detailed results for (-1 for all)')

    args = parser.parse_args()
    main(model_dir=args.model_dir, data_dir=args.data_dir, k=args.k, num_users=args.num_users)