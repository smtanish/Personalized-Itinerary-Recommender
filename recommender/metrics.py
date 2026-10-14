"""
metrics.py - Evaluation metrics.

Error metrics (how close are predicted ratings to real ones?)
    RMSE = sqrt(mean((predicted - actual)^2))
    MAE  = mean(|predicted - actual|)

Decision-support / ranking metrics (is the top-K list useful?)
    Precision@K = relevant items in top K / K
    Recall@K    = relevant items in top K / all relevant items
    NDCG@K      = DCG / ideal DCG, where DCG = sum rel_i / log2(i + 1)
                  (rewards putting relevant items near the top)
"""

import numpy as np


def rmse(pred, actual) -> float:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(pred, actual) -> float:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return float(np.mean(np.abs(pred - actual)))


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0
