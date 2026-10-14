"""
evaluate.py - Offline evaluation on historical (simulated) ratings.

Evaluation design: hold-out split
  * for every user, 20% of their ratings are hidden as the TEST set
  * all models are trained on the remaining 80% (TRAIN set)
  * a test place counts as "relevant" if the user rated it 4 or 5

For each user we rank every place they have NOT rated in TRAIN and check how
many of the hidden relevant places appear in the top K.

Run:  python evaluate.py
Writes results/evaluation.csv and results/hybrid_weights.csv
"""

import numpy as np
import pandas as pd

from recommender.data import RESULTS_DIR, load_places, load_ratings
from recommender.hybrid import HybridRecommender, minmax
from recommender.metrics import mae, ndcg_at_k, precision_at_k, recall_at_k, rmse

SEED = 7
TEST_FRACTION = 0.2
K = 5
RELEVANT_RATING = 4

WEIGHT_GRID = [
    (1.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.6, 0.2, 0.2),
    (0.4, 0.3, 0.3),
    (0.2, 0.4, 0.4),
]


def train_test_split(ratings: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    test_idx = []
    for _, group in ratings.groupby("user_id"):
        n_test = max(2, int(round(len(group) * TEST_FRACTION)))
        test_idx.extend(rng.choice(group.index, size=n_test, replace=False))
    test = ratings.loc[test_idx]
    train = ratings.drop(index=test_idx)
    return train, test


def main() -> None:
    places = load_places()
    ratings = load_ratings()
    train, test = train_test_split(ratings)
    model = HybridRecommender(places, train)

    # Popularity baseline = number of "liked" (>=4) ratings in train.
    liked_counts = train[train["rating"] >= RELEVANT_RATING]["place_id"].value_counts()
    popularity = places["place_id"].map(liked_counts).fillna(0).values

    err = {"User mean (baseline)": ([], []), "Item-based CF": ([], []), "SVD": ([], [])}
    methods = ["Popularity (baseline)", "Content-based", "Item-based CF", "SVD", "Hybrid (0.4/0.3/0.3)"]
    rank = {m: {"P": [], "R": [], "N": []} for m in methods}
    grid = {w: {"P": [], "R": [], "N": []} for w in WEIGHT_GRID}

    pid_list = list(places["place_id"])
    for user_id, user_test in test.groupby("user_id"):
        scores = model.score_all(user_id)
        idx = {pid: i for i, pid in enumerate(pid_list)}
        test_rows = [idx[p] for p in user_test["place_id"]]
        actual = user_test["rating"].values

        # ---- error metrics ----
        user_mean = model.user_ratings(user_id).mean()
        err["User mean (baseline)"][0].extend([user_mean] * len(actual))
        err["Item-based CF"][0].extend(scores["item_cf"].values[test_rows])
        err["SVD"][0].extend(scores["svd"].values[test_rows])
        for v in err.values():
            v[1].extend(actual)

        # ---- ranking metrics ----
        relevant = set(user_test.loc[user_test["rating"] >= RELEVANT_RATING, "place_id"])
        if not relevant:
            continue
        seen = set(model.user_ratings(user_id).index)
        cand = scores[~scores["place_id"].isin(seen)].reset_index(drop=True)
        cand_pop = popularity[[idx[p] for p in cand["place_id"]]]
        c_n, i_n, s_n = minmax(cand["content"].values), minmax(cand["item_cf"].values), minmax(cand["svd"].values)

        def ranked_by(values):
            return list(cand["place_id"].values[np.argsort(-values, kind="stable")])

        method_scores = {
            "Popularity (baseline)": cand_pop.astype(float),
            "Content-based": cand["content"].values,
            "Item-based CF": cand["item_cf"].values,
            "SVD": cand["svd"].values,
            "Hybrid (0.4/0.3/0.3)": 0.4 * c_n + 0.3 * i_n + 0.3 * s_n,
        }
        for m, vals in method_scores.items():
            r = ranked_by(vals)
            rank[m]["P"].append(precision_at_k(r, relevant, K))
            rank[m]["R"].append(recall_at_k(r, relevant, K))
            rank[m]["N"].append(ndcg_at_k(r, relevant, K))
        for w in WEIGHT_GRID:
            r = ranked_by(w[0] * c_n + w[1] * i_n + w[2] * s_n)
            grid[w]["P"].append(precision_at_k(r, relevant, K))
            grid[w]["R"].append(recall_at_k(r, relevant, K))
            grid[w]["N"].append(ndcg_at_k(r, relevant, K))

    # One row per method. Rating-error columns only apply to methods that
    # predict a rating; ranking columns only to methods that produce a list.
    rows = []
    for m in ["User mean (baseline)"] + methods:
        row = {"method": m}
        if m in err:
            row["RMSE"] = round(rmse(*err[m]), 3)
            row["MAE"] = round(mae(*err[m]), 3)
        if m in rank:
            row[f"Precision@{K}"] = round(np.mean(rank[m]["P"]), 3)
            row[f"Recall@{K}"] = round(np.mean(rank[m]["R"]), 3)
            row[f"NDCG@{K}"] = round(np.mean(rank[m]["N"]), 3)
        rows.append(row)
    results = pd.DataFrame(rows)

    grid_rows = [{"content": w[0], "item_cf": w[1], "svd": w[2],
                  f"Precision@{K}": round(np.mean(v["P"]), 3),
                  f"Recall@{K}": round(np.mean(v["R"]), 3),
                  f"NDCG@{K}": round(np.mean(v["N"]), 3)} for w, v in grid.items()]
    grid_df = pd.DataFrame(grid_rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "evaluation.csv", index=False)
    grid_df.to_csv(RESULTS_DIR / "hybrid_weights.csv", index=False)

    n_users = test["user_id"].nunique()
    print(f"Train ratings: {len(train)}  Test ratings: {len(test)}  Users: {n_users}")
    print(f"SVD keeps {model.svd.n_factors} factors = "
          f"{model.svd.explained_variance:.0%} of the variance\n")
    print("Model comparison")
    print(results.to_string(index=False, na_rep="-"))
    print("\nHybrid weight comparison (content / item-CF / SVD)")
    print(grid_df.to_string(index=False))


if __name__ == "__main__":
    main()
