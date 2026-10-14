# Technical README

Developer documentation for the Personalized Itinerary Recommender. For setup and a project overview, see `README.md`.

---

## 1. Architecture

```
                  places.csv        ratings.csv
                      │                  │
                      ▼                  ▼
           ┌──────────────────┐  ┌────────────────────┐
           │ ContentBased     │  │ rating_matrix()    │  users × places, NaN = unrated
           │ TF-IDF on tags   │  └─────────┬──────────┘
           └────────┬─────────┘     ┌──────┴───────┐
                    │               ▼              ▼
                    │         ┌──────────┐   ┌───────────┐
                    │         │ ItemKNN  │   │ SVD (k=8) │
                    │         └────┬─────┘   └─────┬─────┘
                    ▼              ▼               ▼
              ┌──────────────────────────────────────────┐
              │ HybridRecommender.recommend()            │
              │  min-max each → weighted sum             │
              │  no ratings → switch to content only     │
              └───────────────────┬──────────────────────┘
                                  ▼ ranked places + reasons
              ┌──────────────────────────────────────────┐
              │ itinerary.plan_trip()  (constraints)     │
              └───────────────────┬──────────────────────┘
                                  ▼
                           app.py (Streamlit)
```

All models are fitted once when the app starts (`@st.cache_resource`). Fitting the full dataset takes under a second.

## 2. Data schema

### `data/places.csv` (63 rows: 31 Jaipur, 32 Delhi)

| column | type | meaning |
|---|---|---|
| `place_id` | str | `J01`…`J31`, `D01`…`D32` |
| `name`, `city`, `category` | str | display fields; `city` ∈ {Jaipur, Delhi} |
| `tags` | str | space-separated tags, the item features for content-based filtering. Hyphenated tags (`street-food`) are one token |
| `cost_inr` | float | approximate entry fee for an Indian adult, or a typical spend for food/nightlife places |
| `duration_hrs` | float | typical visit length |
| `open_hour`, `close_hour` | float | decimal hours (9.5 = 09:30). One window per day |
| `latitude`, `longitude` | float | WGS84 |
| `description` | str | one-line description (display only) |

### `data/users.csv`
`user_id` (`U001`…`U300`), `personas` (the hidden ground truth used by the simulator, shown in the UI for demos only and never read by any model).

### `data/ratings.csv`
`user_id`, `place_id`, `rating` (int 1–5). 5,231 ratings, 72.3% sparse.

## 3. Data simulation (`generate_data.py`)

Seeded (`SEED = 42`), so it is fully reproducible.

* 9 personas, each a set of liked tags. Every user gets 1 persona (40%) or 2 personas (60%).
* `affinity(u, i) = |tags_i ∩ liked_u| / |tags_i|`
* Hidden place quality `q_i ~ N(0, 0.35)`, plus 0.3 for UNESCO sites.
* Visits: 10–25 per user, sampled without replacement with `P(i) ∝ exp(2.5·affinity + q_i)`. This creates realistic selection bias: people visit what they like.
* `rating = clip(round(2.0 + 2.8·affinity + q_i + b_u + ε), 1, 5)` with `b_u ~ N(0, 0.3)` and `ε ~ N(0, 0.6)`.

Re-running it overwrites `users.csv` and `ratings.csv`. Re-run `evaluate.py` afterwards.

## 4. Algorithms

### 4.1 Content-based (`recommender/content_based.py`)

* **Item profile**: `TfidfVectorizer(token_pattern=r"[^\s]+")` on `tags`. Rows are L2-normalised.
* **User profile**:
  * interests → `vectorizer.transform(" ".join(interests))`
  * ratings → `Σ (r_ui − r̄_u) · v_i` over the places the user rated. If every rating equals the mean, the raw ratings are used instead.
  * If both are present, each part is scaled to unit length and they are added.
* **Score**: `cosine(profile, v_i)`
* `top_profile_tags()` returns the strongest positive profile tags. It is used for the "matches your interests" explanation.

### 4.2 Item-based kNN (`collaborative.py → ItemKNN`)

* Mean-centre each user's ratings; unrated cells become 0.
* `sim = Cᵀ C / (‖c_i‖ ‖c_j‖)` (adjusted cosine), with the diagonal zeroed.
* Prediction for item *i*: take the items the user rated that have `sim > 0`, keep the top *k* = 10, then
  `r̂_ui = r̄_u + Σ s_ij (r_uj − r̄_u) / Σ |s_ij|`. If there is no positive neighbour, return `r̄_u`. Clipped to [1, 5].
* `predict_vector(row, mean)` works for any rating vector, so a new user who rates a few places could be served without refitting. The UI does not use this yet.

### 4.3 SVD (`collaborative.py → SVDRecommender`)

1. Missing cells are filled with the user's mean (in centred form, 0).
2. `U, S, Vᵀ = np.linalg.svd(centred)`
3. Keep `n_factors = 8`, which captures about 36% of the variance.
4. `R̂ = U_k S_k V_kᵀ + r̄_u`, clipped to [1, 5].

This is the "impute then truncate" SVD from the textbook, not Funk-SVD/SGD. It is simpler to explain and good enough at this scale. ALS would be a drop-in alternative inside `fit()`.

### 4.4 Hybrid (`recommender/hybrid.py`)

* `score_all(user_id, interests)` returns raw component scores for all 63 places.
* `recommend(...)`:
  1. Filter by city and, optionally, drop places the user already rated.
  2. Min-max scale each component **within the candidate set** to [0, 1]. A constant vector becomes 0.5.
  3. **Weighted**: `score = w_c·content + w_i·item_cf + w_s·svd` (weights renormalised to sum to 1). Default is 0.4 / 0.3 / 0.3, chosen by the grid in `results/hybrid_weights.csv`.
  4. **Switching**: an unknown user, or `user_id=None`, gets weights 1 / 0 / 0.
  5. Adds a `reason` string and sorts by score.
* The result DataFrame carries `attrs["mode"]` and `attrs["weights"]`.

### 4.5 Itinerary planner (`recommender/itinerary.py`)

Greedy constraint satisfaction, one day at a time:

```
pos ← city centre, clock ← start_hour, spent ← 0
while visits < max_stops(pace):
    if clock ≥ 13:00 and no lunch yet: insert 45-min lunch (if clock ≤ 15:00)
    for each unused candidate:
        km      = haversine(pos, place)
        arrive  = clock + (10 + 1.3·km / 20 km/h · 60) min
        if arrive < open: wait if ≤ 1 h, else skip
        depart  = arrive + duration
        feasible if depart ≤ close, depart ≤ end_hour, spent + cost ≤ daily_budget
        utility = score − 0.02 · km
    choose max utility; else end the day
```

Tunable constants are at the top of the file: `PACE_STOPS`, `DISTANCE_PENALTY`, `ROAD_FACTOR`, `CITY_SPEED_KMPH`, `BUFFER_MIN`, `MAX_WAIT_HRS`, `LUNCH_*`. City start points are in `data.CITY_CENTRES`.

Known simplifications: greedy is not globally optimal (day 1 gets the best places), each day restarts at the city centre with no return leg, and there is a single opening window with no weekday closures.

## 5. Evaluation protocol (`evaluate.py`)

* Per-user random hold-out: 20% of ratings (at least 2) go to test. `SEED = 7`.
* All models are refit on train only.
* **Error metrics** on every test rating: RMSE and MAE for item-CF, SVD and the user-mean baseline.
* **Ranking metrics**, K = 5: candidates are all places not in the user's train set. Relevant means the test rating is ≥ 4. Users with no relevant test item (49 of 300) are skipped. The script reports Precision@5, Recall@5 and NDCG@5 for popularity, content, item-CF, SVD, hybrid, and a hybrid weight grid.
* Outputs go to `results/evaluation.csv` and `results/hybrid_weights.csv`. The Evaluation tab in the app reads these.

Caveat: ratings are simulated from tags, so content-based filtering has an advantage over what it would get on real data. The hybrid still wins, and that conclusion is the point.

## 6. Module API cheat sheet

```python
from recommender.data import load_places, load_ratings
from recommender.hybrid import HybridRecommender
from recommender.itinerary import plan_trip, trip_to_frame

model = HybridRecommender(load_places(), load_ratings(), n_factors=8, k_neighbours=10)
recs  = model.recommend(user_id="U003", interests=["fort"], city="Jaipur",
                        weights={"content": .4, "item_cf": .3, "svd": .3})
trip  = plan_trip(recs, "Jaipur", days=3, start_hour=9, end_hour=20,
                  daily_budget=1500, pace="Balanced", lunch=True)
print(trip_to_frame(trip))
```

## 7. Extending

* **Add a place**: append a row to `places.csv` with a new `place_id`, then re-run `generate_data.py` so it gets ratings.
* **Add a city**: add places, then add its centre to `CITY_CENTRES` in `data.py`. The UI picks up new cities automatically.
* **Add a model**: produce a score array in `score_all`, add a weight key, and add it to `recommend` and `evaluate.py`.
* **Use real data**: replace `ratings.csv` with real `user_id, place_id, rating` rows. Nothing else changes.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate the venv, or run `python -m streamlit run app.py` |
| `ModuleNotFoundError: recommender` | Run commands from the project root folder |
| A day shows "Nothing else fits" | Increase the budget or time window, or reduce days. The city may have run out of matching places |
| Evaluation tab is empty | Run `python evaluate.py` or press the button in the tab |

Tested with Python 3.13, Streamlit 1.65, pandas 3.0, NumPy 2.5 and scikit-learn 1.9.
