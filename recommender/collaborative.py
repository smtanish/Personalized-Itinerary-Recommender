"""
collaborative.py - Collaborative filtering on the user x place rating matrix.

ItemKNN  : memory-based, item-based nearest neighbours.
           similarity(i, j) = adjusted cosine (ratings are mean-centred per
           user before taking cosine between item columns).
           prediction r(u, i) = mean_u + sum_j sim(i,j) * (r(u,j) - mean_u)
                                         / sum_j |sim(i,j)|
           over the k most similar places j that user u has rated.

SVDRecommender : model-based matrix factorisation.
           1. fill each missing rating with the user's mean (missing values)
           2. subtract the user mean (normalisation)
           3. truncated SVD keeps only k latent factors (dimensionality reduction)
           4. reconstruct U_k * S_k * V_k^T + mean  -> predicted ratings
"""

import numpy as np
import pandas as pd

RATING_MIN, RATING_MAX = 1.0, 5.0


def _user_means(R: np.ndarray) -> np.ndarray:
    """Row means ignoring NaN; users with no ratings get the global mean."""
    counts = (~np.isnan(R)).sum(axis=1)
    sums = np.nansum(R, axis=1)
    global_mean = np.nanmean(R)
    return np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)


class ItemKNN:
    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, matrix: pd.DataFrame) -> "ItemKNN":
        self.items = list(matrix.columns)
        self.users = {u: i for i, u in enumerate(matrix.index)}
        self.R = matrix.values.astype(float)
        self.means = _user_means(self.R)

        centred = self.R - self.means[:, None]
        centred[np.isnan(centred)] = 0.0
        norms = np.linalg.norm(centred, axis=0)
        denom = np.outer(norms, norms)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, centred.T @ centred / denom, 0.0)
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        return self

    def predict_vector(self, user_row: np.ndarray, user_mean: float) -> np.ndarray:
        """Predict every item for one user given their (NaN-padded) rating row."""
        rated = np.where(~np.isnan(user_row))[0]
        preds = np.full(len(self.items), user_mean)
        if rated.size == 0:
            return preds
        deviations = user_row[rated] - user_mean
        for i in range(len(self.items)):
            sims = self.sim[i, rated]
            positive = sims > 0
            if not positive.any():
                continue
            s, d = sims[positive], deviations[positive]
            top = np.argsort(s)[::-1][: self.k]
            preds[i] = user_mean + (s[top] @ d[top]) / np.abs(s[top]).sum()
        return np.clip(preds, RATING_MIN, RATING_MAX)

    def predict_user(self, user_id: str) -> np.ndarray:
        u = self.users[user_id]
        return self.predict_vector(self.R[u], self.means[u])

    def most_similar(self, place_id: str, n: int = 5) -> list:
        i = self.items.index(place_id)
        order = np.argsort(self.sim[i])[::-1][:n]
        return [(self.items[j], float(self.sim[i, j])) for j in order]


class SVDRecommender:
    def __init__(self, n_factors: int = 8):
        self.n_factors = n_factors

    def fit(self, matrix: pd.DataFrame) -> "SVDRecommender":
        self.items = list(matrix.columns)
        self.users = {u: i for i, u in enumerate(matrix.index)}
        R = matrix.values.astype(float)
        means = _user_means(R)

        centred = R - means[:, None]
        centred[np.isnan(centred)] = 0.0          # missing = user's mean rating

        U, S, Vt = np.linalg.svd(centred, full_matrices=False)
        k = self.n_factors
        self.user_factors = U[:, :k] * S[:k]      # users in latent space
        self.item_factors = Vt[:k, :].T           # places in latent space
        self.singular_values = S
        self.explained_variance = float((S[:k] ** 2).sum() / (S ** 2).sum())

        approx = self.user_factors @ self.item_factors.T + means[:, None]
        self.predictions = np.clip(approx, RATING_MIN, RATING_MAX)
        return self

    def predict_user(self, user_id: str) -> np.ndarray:
        return self.predictions[self.users[user_id]]
