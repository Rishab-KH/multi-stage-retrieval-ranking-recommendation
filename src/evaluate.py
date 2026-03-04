import math
from typing import List, Dict, Iterable, Callable


def recall_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    gt = set(ground_truth)
    if len(gt) == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in gt)
    return hits / len(gt)


def ndcg_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    gt = set(ground_truth)
    if len(gt) == 0:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in gt:
            dcg += 1.0 / math.log2(i + 2)

    # ideal DCG: all gt items ranked at top
    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    gt = set(ground_truth)
    for i, item in enumerate(recommended[:k]):
        if item in gt:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_recommendations(recs: Dict[int, List[int]],
                             truths: Dict[int, List[int]],
                             ks=(10, 20)) -> Dict[str, float]:
    """
    recs: mapping user_idx -> list of recommended product_idx
    truths: mapping user_idx -> list of ground truth product_idx
    """
    results = {}
    for k in ks:
        recalls = []
        ndcgs = []
        mrrs = []
        for u, recommended in recs.items():
            ground = truths.get(u, [])
            recalls.append(recall_at_k(recommended, ground, k))
            ndcgs.append(ndcg_at_k(recommended, ground, k))
            mrrs.append(mrr_at_k(recommended, ground, k))

        results[f'Recall@{k}'] = float(sum(recalls) / len(recalls))
        results[f'NDCG@{k}'] = float(sum(ndcgs) / len(ndcgs))
        results[f'MRR@{k}'] = float(sum(mrrs) / len(mrrs))
    return results


def evaluate_orderable_recommendations(
    recs: Dict[int, List[int]],
    truths: Dict[int, List[int]],
    is_orderable: Callable[[int, int], bool],
    ks=(10, 20),
) -> Dict[str, float]:
    """
    Inventory-aware evaluation.

    Metrics:
      - OrderableRecall@K: hits among orderable recommended items divided by |ground_truth|
      - OrderableRate@K: fraction of top-K recommendations that are orderable
    """
    results = {}
    for k in ks:
        orderable_recalls = []
        orderable_rates = []
        for u, recommended in recs.items():
            ground = set(truths.get(u, []))
            rec_k = recommended[:k]

            if len(ground) == 0:
                orderable_recalls.append(0.0)
                orderable_rates.append(0.0)  # no ground truth → rate is undefined, default 0
                continue

            orderable_rec_k = [it for it in rec_k if is_orderable(it, u)]
            hits = sum(1 for it in orderable_rec_k if it in ground)

            orderable_recalls.append(hits / len(ground))
            denom = len(rec_k) if len(rec_k) > 0 else 1
            orderable_rates.append(len(orderable_rec_k) / denom)

        results[f'OrderableRecall@{k}'] = float(sum(orderable_recalls) / len(orderable_recalls))
        results[f'OrderableRate@{k}'] = float(sum(orderable_rates) / len(orderable_rates))

    return results