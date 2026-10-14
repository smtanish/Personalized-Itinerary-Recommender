"""
generate_data.py
----------------
Creates the simulated users and ratings used by the recommender.

Why simulate?  Real travel-rating data for specific Jaipur/Delhi attractions is
not publicly available in a clean form, so we create believable users:

  1. Every user is given 1-2 "travel personas" (e.g. heritage buff + foodie).
     Each persona is a set of tags the user enjoys.
  2. Each place gets a small hidden "quality" bonus (famous UNESCO sites get a
     bit more) so that some places are liked by almost everyone.
  3. A user visits 10-25 places.  They are more likely to visit places that
     match their interests or are popular (just like real tourists).
  4. The rating for a visited place is:
        rating = 2.0 + 2.8 * affinity + quality + user_bias + noise
     rounded and clipped to 1..5, where affinity = share of the place's tags
     that the user likes.

The personas are saved in users.csv only so humans can sanity-check results.
The recommender NEVER reads them - it only sees the ratings.

Run:  python generate_data.py
"""

import numpy as np
import pandas as pd

from recommender.data import DATA_DIR, load_places

SEED = 42
N_USERS = 300
MIN_VISITS, MAX_VISITS = 10, 25

PERSONAS = {
    "heritage buff": {"heritage", "history", "fort", "palace", "architecture", "unesco", "museum"},
    "foodie": {"food", "street-food", "cafes", "culture"},
    "shopper": {"shopping", "handicrafts", "culture"},
    "nature lover": {"nature", "garden", "wildlife", "viewpoint", "adventure"},
    "spiritual seeker": {"spiritual", "temple", "culture", "architecture"},
    "night owl": {"nightlife", "cafes", "entertainment", "food"},
    "photographer": {"photography", "viewpoint", "architecture", "art"},
    "family traveller": {"family", "museum", "science", "wildlife", "entertainment", "garden"},
    "art & culture fan": {"art", "culture", "museum", "handicrafts"},
}


def affinity(place_tags: set, liked_tags: set) -> float:
    """Fraction of a place's tags that the user likes (0..1)."""
    if not place_tags:
        return 0.0
    return len(place_tags & liked_tags) / len(place_tags)


def main() -> None:
    rng = np.random.default_rng(SEED)
    places = load_places()
    place_tags = [set(t.split()) for t in places["tags"]]

    # Hidden per-place quality: small random effect + bonus for UNESCO sites.
    quality = rng.normal(0.0, 0.35, len(places))
    quality += np.array([0.3 if "unesco" in t else 0.0 for t in place_tags])

    persona_names = list(PERSONAS)
    users, ratings = [], []

    for u in range(1, N_USERS + 1):
        user_id = f"U{u:03d}"
        k = rng.choice([1, 2], p=[0.4, 0.6])
        chosen = list(rng.choice(persona_names, size=k, replace=False))
        liked = set().union(*(PERSONAS[p] for p in chosen))
        user_bias = rng.normal(0.0, 0.3)

        aff = np.array([affinity(t, liked) for t in place_tags])

        # Probability of visiting: interest + popularity, softmax-style.
        visit_weight = np.exp(2.5 * aff + quality)
        visit_prob = visit_weight / visit_weight.sum()
        n_visits = rng.integers(MIN_VISITS, MAX_VISITS + 1)
        visited = rng.choice(len(places), size=n_visits, replace=False, p=visit_prob)

        for idx in visited:
            raw = 2.0 + 2.8 * aff[idx] + quality[idx] + user_bias + rng.normal(0.0, 0.6)
            rating = int(np.clip(np.rint(raw), 1, 5))
            ratings.append((user_id, places.loc[idx, "place_id"], rating))

        users.append((user_id, " + ".join(chosen)))

    users_df = pd.DataFrame(users, columns=["user_id", "personas"])
    ratings_df = pd.DataFrame(ratings, columns=["user_id", "place_id", "rating"])

    users_df.to_csv(DATA_DIR / "users.csv", index=False)
    ratings_df.to_csv(DATA_DIR / "ratings.csv", index=False)

    density = len(ratings_df) / (len(users_df) * len(places))
    print(f"Users: {len(users_df)}  Places: {len(places)}  Ratings: {len(ratings_df)}")
    print(f"Matrix density: {density:.1%} (sparsity {1 - density:.1%})")
    print("Rating distribution:")
    print(ratings_df["rating"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
