# Personalized Itinerary Recommender

Mini project for **Recommender Systems (AIM3143)**, B.Tech CSE (AIML), Manipal University Jaipur, Jul–Nov 2026.

| | |
|---|---|
| Team | [Name 1] ([Reg No.]) · [Name 2] ([Reg No.]) |
| Instructor | [Instructor name] |

## What it does

You tell the app who you are, which city you're visiting (Jaipur or Delhi), how many days (1–5), your daily time window, budget and pace. It:

1. **Ranks every attraction for you** with a hybrid recommender (content-based + item-based collaborative filtering + SVD matrix factorisation).
2. **Builds a day-by-day itinerary** with a constraint-based planner that respects opening hours, travel time, your daily budget and pace.
3. **Explains every pick** ("matches your interests: fort, heritage; travellers with similar taste rate it ~4.3/5").

A new traveller with no history gets recommendations from their interests alone (the cold-start case). An existing traveller gets the full hybrid.

## Quick start (VS Code)

```bash
# 1. open the folder in VS Code, then in the terminal:
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. (optional) regenerate simulated users/ratings — already included in data/
python generate_data.py

# 3. (optional) run the offline evaluation — results already included in results/
python evaluate.py

# 4. launch the app (opens in your browser at http://localhost:8501)
streamlit run app.py
```

Requires Python 3.9 or newer.

## Using the app

* **Sidebar → Traveller**: choose an *existing traveller* (one of 300 simulated users) or a *new traveller*, then pick interests.
* **Sidebar → Trip**: city, days, time window, budget per day, pace, lunch break.
* **Itinerary tab**: the plan for each day as a table plus a map.
* **Recommendations tab**: top 15 places with each model's score and a chart showing how the hybrid score is built.
* **Evaluation tab**: model comparison and hybrid-weight tuning. Press *Run evaluation now* to recompute.
* **How it works tab**: the pipeline in plain words.

## Results (hold-out test, 20% of each user's ratings hidden)

| Method | RMSE ↓ | MAE ↓ | Precision@5 ↑ | Recall@5 ↑ | NDCG@5 ↑ |
|---|---|---|---|---|---|
| User mean (baseline) | 1.081 | 0.886 | – | – | – |
| Popularity (baseline) | – | – | 0.096 | 0.234 | 0.171 |
| Content-based | – | – | 0.202 | 0.528 | 0.440 |
| Item-based CF | **0.893** | **0.705** | 0.171 | 0.440 | 0.333 |
| SVD (8 factors) | 0.954 | 0.772 | 0.193 | 0.510 | 0.429 |
| **Hybrid (0.4 / 0.3 / 0.3)** | – | – | **0.228** | **0.584** | **0.484** |

The hybrid beats every single model on all three ranking metrics. Precision@5 looks low only because each user has about 2 hidden "liked" places, so 5 recommendations can be at most ~40% precise.

## How this maps to the syllabus

| Syllabus topic | Where in the project |
|---|---|
| Explicit ratings, sparsity, cold start | `data/ratings.csv` (72% sparse), cold-start switching in `hybrid.py` |
| Matrix representation, cosine similarity | `data.py` (rating matrix), `content_based.py`, `collaborative.py` |
| Item-based nearest-neighbour CF | `collaborative.py → ItemKNN` |
| Matrix factorisation (SVD), latent factors | `collaborative.py → SVDRecommender` |
| Preprocessing: normalisation, missing values, dimensionality reduction | mean-centring, mean-imputation, truncated SVD |
| Content-based: item profiles, user profiles, similarity retrieval | `content_based.py` (TF-IDF tag profiles) |
| Knowledge-based: constraint-based recommendation | `itinerary.py` (hours, budget, time, pace) |
| Hybrid: weighted (parallel) and switching | `hybrid.py` |
| Evaluation on historical data: error and decision-support metrics | `evaluate.py`, `metrics.py` |

## Project structure

```
itinerary-recommender/
├── app.py                  Streamlit user interface
├── generate_data.py        creates simulated users + ratings
├── evaluate.py             offline evaluation
├── recommender/
│   ├── data.py             loading CSVs, rating matrix
│   ├── content_based.py    TF-IDF + cosine similarity
│   ├── collaborative.py    item-based kNN and SVD
│   ├── hybrid.py           weighted + switching hybrid
│   ├── itinerary.py        constraint-based day planner
│   └── metrics.py          RMSE, MAE, Precision/Recall/NDCG@K
├── data/                   places.csv, users.csv, ratings.csv
├── results/                evaluation outputs
├── presentation/           slides (.pptx, 15 slides with speaker notes)
├── README.md               this file
└── TECHNICAL_README.md     developer documentation
```

## Data note

Attractions, locations, typical durations, opening hours and costs are real places, but the figures are **approximate** and simplified for an academic project (one opening window per day, no weekly closures). Users and ratings are **simulated** (see `generate_data.py`) because no public rating dataset exists for these exact places.

## References

1. C.C. Aggarwal, *Recommender Systems: The Textbook*, Springer, 2016.
2. F. Ricci, L. Rokach, B. Shapira, P.B. Kantor, *Recommender Systems Handbook*, Springer, 2011.
3. N. Manouselis, H. Drachsler, K. Verbert, E. Duval, *Recommender Systems for Learning*, Springer, 2013.
