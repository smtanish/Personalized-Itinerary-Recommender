"""
data.py - loading the CSV files and building the user-item rating matrix.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

# Starting point of every day (treated as the traveller's hotel).
CITY_CENTRES = {
    "Jaipur": (26.9239, 75.8267),   # near Hawa Mahal, old city
    "Delhi": (28.6315, 77.2167),    # Connaught Place
}


def load_places() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "places.csv")


def load_users() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "users.csv")


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ratings.csv")


def all_tags(places: pd.DataFrame) -> list:
    """Sorted list of every tag in the dataset (used for the interest picker)."""
    tags = set()
    for t in places["tags"]:
        tags.update(t.split())
    return sorted(tags)


def rating_matrix(ratings: pd.DataFrame, places: pd.DataFrame, users=None) -> pd.DataFrame:
    """
    Users x places matrix. Missing ratings are NaN.
    Columns always follow the order of places.csv so every module agrees on
    which column is which place.
    """
    matrix = ratings.pivot_table(index="user_id", columns="place_id", values="rating")
    matrix = matrix.reindex(columns=places["place_id"])
    if users is not None:
        matrix = matrix.reindex(index=users)
    return matrix


def sparsity(matrix: pd.DataFrame) -> float:
    """Share of the matrix that is empty (0..1)."""
    return float(np.isnan(matrix.values).mean())
