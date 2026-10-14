"""
hybrid.py - Combines the three recommenders.

Two hybridisation designs from the syllabus are used:

* Weighted (parallel) hybrid
      score = w_content * content + w_itemcf * item_cf + w_svd * svd
  Each component is min-max scaled to 0..1 first so they are comparable.

* Switching hybrid (cold start)
  A new traveller has no ratings, so collaborative filtering cannot say
  anything about them.  In that case the system switches to content-based
  only, using the interests they picked.
"""

import numpy as np
import pandas as pd

from .collaborative import ItemKNN, SVDRecommender
from .content_based import ContentBasedRecommender
from .data import rating_matrix

DEFAULT_WEIGHTS = {"content": 0.4, "item_cf": 0.3, "svd": 0.3}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


class HybridRecommender:
    def __init__(self, places: pd.DataFrame, ratings: pd.DataFrame,
                 n_factors: int = 8, k_neighbours: int = 10):
        self.places = places.reset_index(drop=True)
        self.ratings = ratings
        self.matrix = rating_matrix(ratings, self.places)

        self.content = ContentBasedRecommender(self.places)
        self.item_cf = ItemKNN(k=k_neighbours).fit(self.matrix)
        self.svd = SVDRecommender(n_factors=n_factors).fit(self.matrix)

    def known_user(self, user_id) -> bool:
        return user_id is not None and user_id in self.matrix.index

    def user_ratings(self, user_id) -> pd.Series:
        if not self.known_user(user_id):
            return pd.Series(dtype=float)
        return self.matrix.loc[user_id].dropna()

    def score_all(self, user_id=None, interests=()) -> pd.DataFrame:
        """
        Raw component scores for every place (no filtering, no weighting).
        content : cosine similarity (0..1)
        item_cf : predicted rating (1..5) or NaN for a new user
        svd     : predicted rating (1..5) or NaN for a new user
        """
        known = self.known_user(user_id)
        ur = self.user_ratings(user_id) if known else None
        profile = self.content.build_profile(interests, ur)

        out = self.places.copy()
        out["content"] = self.content.score(profile)
        if known:
            out["item_cf"] = self.item_cf.predict_user(user_id)
            out["svd"] = self.svd.predict_user(user_id)
        else:
            out["item_cf"] = np.nan
            out["svd"] = np.nan
        out.attrs["profile_tags"] = self.content.top_profile_tags(profile)
        out.attrs["mode"] = "weighted hybrid" if known else "cold start: content-based only"
        return out

    def recommend(self, user_id=None, interests=(), weights=None, city=None,
                  exclude_visited=True, top_n=None) -> pd.DataFrame:
        weights = dict(weights or DEFAULT_WEIGHTS)
        scores = self.score_all(user_id, interests)
        known = self.known_user(user_id)

        if city:
            scores = scores[scores["city"] == city]
        if known and exclude_visited:
            scores = scores[~scores["place_id"].isin(self.user_ratings(user_id).index)]
        scores = scores.copy()
        if scores.empty:
            return scores

        # Switching: no ratings -> only the content component is usable.
        if not known:
            weights = {"content": 1.0, "item_cf": 0.0, "svd": 0.0}
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

        scores["content_n"] = minmax(scores["content"].values)
        scores["item_cf_n"] = minmax(scores["item_cf"].fillna(0).values) if known else 0.0
        scores["svd_n"] = minmax(scores["svd"].fillna(0).values) if known else 0.0
        scores["score"] = (weights["content"] * scores["content_n"]
                           + weights["item_cf"] * scores["item_cf_n"]
                           + weights["svd"] * scores["svd_n"])

        profile_tags = set(scores.attrs.get("profile_tags", []))
        scores["reason"] = [self._reason(row, profile_tags, known) for row in scores.itertuples()]
        scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
        scores.attrs["weights"] = weights
        scores.attrs["mode"] = "weighted hybrid" if known else "cold start: content-based only"
        return scores.head(top_n) if top_n else scores

    @staticmethod
    def _reason(row, profile_tags: set, known: bool) -> str:
        parts = []
        matched = [t for t in row.tags.split() if t in profile_tags]
        if matched:
            parts.append("matches your interests: " + ", ".join(matched[:3]))
        if known and not np.isnan(row.item_cf) and row.item_cf >= 4.0:
            parts.append(f"travellers with similar taste rate it ~{row.item_cf:.1f}/5")
        elif known and not np.isnan(row.svd) and row.svd >= 4.0:
            parts.append(f"predicted rating {row.svd:.1f}/5 from your taste pattern")
        return "; ".join(parts).capitalize() if parts else "Popular, well-rated stop"
