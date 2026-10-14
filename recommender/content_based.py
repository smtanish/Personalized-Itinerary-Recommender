"""
content_based.py - Content-based filtering.

Item profile : TF-IDF vector of each place's tags (e.g. "fort heritage unesco").
User profile : built from
               (a) interests the traveller ticks in the app, and/or
               (b) places they rated - each rated place's vector is weighted by
                   (rating - user's mean rating), so liked places pull the
                   profile towards them and disliked places push it away.
Score        : cosine similarity between the user profile and each item profile.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class ContentBasedRecommender:
    def __init__(self, places: pd.DataFrame):
        self.places = places
        self.place_index = {pid: i for i, pid in enumerate(places["place_id"])}
        # Split tags on whitespace only so "street-food" stays one token.
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.item_matrix = self.vectorizer.fit_transform(places["tags"])  # rows are L2-normalised
        self.vocab = np.array(self.vectorizer.get_feature_names_out())

    # ---------- user profiles ----------
    def profile_from_interests(self, interests) -> np.ndarray:
        if not interests:
            return np.zeros(len(self.vocab))
        vec = self.vectorizer.transform([" ".join(interests)]).toarray().ravel()
        return vec

    def profile_from_ratings(self, user_ratings: pd.Series) -> np.ndarray:
        """user_ratings: Series indexed by place_id with the user's ratings."""
        user_ratings = user_ratings.dropna()
        if user_ratings.empty:
            return np.zeros(len(self.vocab))
        weights = user_ratings - user_ratings.mean()
        if np.allclose(weights, 0):          # user gave every place the same rating
            weights = user_ratings
        rows = [self.place_index[pid] for pid in user_ratings.index]
        profile = self.item_matrix[rows].T @ weights.values
        return np.asarray(profile).ravel()

    def build_profile(self, interests=(), user_ratings=None) -> np.ndarray:
        """Combine interest-based and rating-based profiles (each unit length)."""
        parts = [self.profile_from_interests(interests)]
        if user_ratings is not None:
            parts.append(self.profile_from_ratings(user_ratings))
        profile = np.zeros(len(self.vocab))
        for p in parts:
            norm = np.linalg.norm(p)
            if norm > 0:
                profile += p / norm
        return profile

    # ---------- scoring ----------
    def score(self, profile: np.ndarray) -> np.ndarray:
        """Cosine similarity of the profile with every place (order = places.csv)."""
        if not np.any(profile):
            return np.zeros(len(self.places))
        return cosine_similarity(profile.reshape(1, -1), self.item_matrix).ravel()

    def top_profile_tags(self, profile: np.ndarray, n: int = 8) -> list:
        """Tags with the highest positive weight in the profile (for explanations)."""
        order = np.argsort(profile)[::-1]
        return [self.vocab[i] for i in order[:n] if profile[i] > 0]
