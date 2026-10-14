"""
app.py - Streamlit front end for the Personalized Itinerary Recommender.

Run:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from recommender.data import (RESULTS_DIR, all_tags, load_places, load_ratings,
                              load_users, sparsity)
from recommender.hybrid import DEFAULT_WEIGHTS, HybridRecommender
from recommender.itinerary import PACE_STOPS, plan_trip, trip_to_frame

st.set_page_config(page_title="Itinerary Recommender", page_icon="🧭", layout="wide")


@st.cache_resource
def get_model():
    return HybridRecommender(load_places(), load_ratings())


@st.cache_data
def get_users():
    users = load_users()
    counts = load_ratings()["user_id"].value_counts()
    users["n_ratings"] = users["user_id"].map(counts).fillna(0).astype(int)
    return users


model = get_model()
places = model.places
users = get_users()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Traveller")
traveller_type = st.sidebar.radio(
    "Who is travelling?",
    ["Existing traveller (has ratings)", "New traveller (no history)"],
)
is_new = traveller_type.startswith("New")

user_id = None
if not is_new:
    labels = {f"{r.user_id} · {r.personas} ({r.n_ratings} ratings)": r.user_id
              for r in users.itertuples()}
    user_id = labels[st.sidebar.selectbox("Pick a user", list(labels))]
    st.sidebar.caption("Personas are shown for the demo only. The model never sees "
                       "them; it learns only from the ratings.")

interests = st.sidebar.multiselect(
    "Interests" + (" (required)" if is_new else " (optional)"),
    all_tags(places),
    default=["heritage", "food"] if is_new else [],
)

st.sidebar.header("Trip")
city = st.sidebar.selectbox("City", sorted(places["city"].unique()), index=0)
days = st.sidebar.slider("Number of days", 1, 5, 2)
start_hour, end_hour = st.sidebar.slider("Daily time window (hours)", 6, 23, (9, 20))
daily_budget = st.sidebar.number_input("Budget per day (₹, entry fees & food stops)",
                                       min_value=0, max_value=20000, value=1500, step=100)
pace = st.sidebar.radio("Pace", list(PACE_STOPS), index=1, horizontal=True,
                        help="Relaxed = 3, Balanced = 4, Packed = 6 places per day")
lunch = st.sidebar.checkbox("Add a lunch break", value=True)
skip_visited = True
if not is_new:
    skip_visited = st.sidebar.checkbox("Skip places this user already rated", value=True)

with st.sidebar.expander("Hybrid weights (advanced)"):
    w_content = st.slider("Content-based", 0.0, 1.0, DEFAULT_WEIGHTS["content"], 0.05)
    w_itemcf = st.slider("Item-based CF", 0.0, 1.0, DEFAULT_WEIGHTS["item_cf"], 0.05)
    w_svd = st.slider("SVD", 0.0, 1.0, DEFAULT_WEIGHTS["svd"], 0.05)
    st.caption("Weights are rescaled to sum to 1. Ignored for new travellers.")

# ------------------------------------------------------------------ compute
st.title("🧭 Personalized Itinerary Recommender")
st.caption("Hybrid recommender (content-based + item-based CF + SVD) with a "
           "constraint-based day planner · Jaipur & Delhi")

if is_new and not interests:
    st.warning("Pick at least one interest in the sidebar so the system knows what you like.")
    st.stop()

weights = {"content": w_content, "item_cf": w_itemcf, "svd": w_svd}
if sum(weights.values()) == 0:
    weights = DEFAULT_WEIGHTS
recs = model.recommend(user_id=user_id, interests=interests, weights=weights,
                       city=city, exclude_visited=skip_visited)
trip = plan_trip(recs, city, days=days, start_hour=start_hour, end_hour=end_hour,
                 daily_budget=daily_budget, pace=pace, lunch=lunch)
plan = trip_to_frame(trip)

mode = recs.attrs.get("mode", "")
if is_new:
    st.info(f"**Mode: {mode}.** A new traveller has no ratings, so collaborative "
            "filtering can't help yet; the system switches to matching your interests.")
else:
    w = recs.attrs.get("weights", weights)
    st.info(f"**Mode: {mode}** · content {w['content']:.2f} · item-CF {w['item_cf']:.2f} "
            f"· SVD {w['svd']:.2f}")

tab_plan, tab_recs, tab_eval, tab_how = st.tabs(
    ["🗓️ Itinerary", "⭐ Recommendations", "📊 Evaluation", "ℹ️ How it works"])

# ------------------------------------------------------------------ itinerary
with tab_plan:
    visits = plan[plan["kind"] == "visit"] if not plan.empty else plan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Places", len(visits))
    c2.metric("Days", days)
    c3.metric("Total cost", f"₹{visits['cost_inr'].sum():,.0f}" if len(visits) else "₹0")
    c4.metric("Travel", f"{visits['travel_km'].sum():.1f} km" if len(visits) else "0 km")

    day_tabs = st.tabs([f"Day {d}" for d in range(1, days + 1)])
    for d, tab in enumerate(day_tabs, start=1):
        with tab:
            day = plan[plan["day"] == d] if not plan.empty else plan
            day_visits = day[day["kind"] == "visit"] if not day.empty else day
            if day_visits.empty:
                st.warning("Nothing else fits this day. Try a bigger budget, a longer "
                           "time window, fewer days, or more interests.")
                continue
            st.dataframe(
                day[["time", "place", "category", "travel_km", "travel_min", "cost_inr", "why"]],
                hide_index=True,
                column_config={
                    "time": "Time", "place": "Place", "category": "Type",
                    "travel_km": st.column_config.NumberColumn("Travel (km)", format="%.1f"),
                    "travel_min": st.column_config.NumberColumn("Travel (min)", format="%d"),
                    "cost_inr": st.column_config.NumberColumn("Cost (₹)", format="%d"),
                    "why": "Why recommended",
                },
            )
            st.map(day_visits[["latitude", "longitude"]], size=60, zoom=11)
            st.caption(f"Day cost ₹{day_visits['cost_inr'].sum():,.0f} of ₹{daily_budget:,} · "
                       "each day starts from the city centre.")

# ------------------------------------------------------------------ recommendations
with tab_recs:
    st.subheader(f"Top places in {city} for this traveller")
    top = recs.head(15).copy()
    top["rank"] = range(1, len(top) + 1)
    show_cols = ["rank", "name", "category", "content", "item_cf", "svd", "score", "reason"]
    st.dataframe(
        top[show_cols], hide_index=True,
        column_config={
            "rank": "#", "name": "Place", "category": "Type",
            "content": st.column_config.NumberColumn("Content (cosine)", format="%.2f"),
            "item_cf": st.column_config.NumberColumn("Item-CF (pred. rating)", format="%.2f"),
            "svd": st.column_config.NumberColumn("SVD (pred. rating)", format="%.2f"),
            "score": st.column_config.ProgressColumn("Hybrid score", min_value=0, max_value=1, format="%.2f"),
            "reason": "Why",
        },
    )

    st.markdown("**How the hybrid score is built (top 10)**: each bar is the "
                "weighted contribution of a component after scaling it to 0–1.")
    w = recs.attrs["weights"]
    contrib = pd.DataFrame({
        "Content-based": w["content"] * top["content_n"],
        "Item-based CF": w["item_cf"] * top["item_cf_n"],
        "SVD": w["svd"] * top["svd_n"],
    }).head(10)
    contrib.index = top["name"].head(10)
    st.bar_chart(contrib)

    if not is_new:
        with st.expander("Places this user has already rated (their history)"):
            hist = model.user_ratings(user_id).rename("rating").reset_index()
            hist = hist.merge(places[["place_id", "name", "city", "tags"]], on="place_id")
            st.dataframe(hist.sort_values("rating", ascending=False), hide_index=True)
        with st.expander("Item-based CF: places most similar to this user's favourite"):
            fav = model.user_ratings(user_id).idxmax()
            sims = pd.DataFrame(model.item_cf.most_similar(fav, 5), columns=["place_id", "similarity"])
            sims = sims.merge(places[["place_id", "name"]], on="place_id")
            st.write(f"Favourite: **{places.set_index('place_id').loc[fav, 'name']}**")
            st.dataframe(sims[["name", "similarity"]], hide_index=True)

# ------------------------------------------------------------------ evaluation
with tab_eval:
    m = model.matrix
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", m.shape[0])
    c2.metric("Places", m.shape[1])
    c3.metric("Ratings", int(m.notna().sum().sum()))
    c4.metric("Sparsity", f"{sparsity(m):.0%}")

    eval_file = RESULTS_DIR / "evaluation.csv"
    grid_file = RESULTS_DIR / "hybrid_weights.csv"
    if st.button("Run evaluation now"):
        import evaluate
        with st.spinner("Training on 80% of ratings and testing on 20%..."):
            evaluate.main()
    if eval_file.exists():
        res = pd.read_csv(eval_file)
        st.subheader("Model comparison (hold-out test, K = 5)")
        st.dataframe(res, hide_index=True)
        ranking = res.dropna(subset=["NDCG@5"]).set_index("method")[["Precision@5", "Recall@5", "NDCG@5"]]
        st.bar_chart(ranking)
        st.caption("RMSE/MAE measure rating-prediction error (lower is better). "
                   "Precision, Recall and NDCG@5 measure the quality of the top-5 list "
                   "(higher is better). A place is 'relevant' if the hidden rating is 4 or 5.")
        if grid_file.exists():
            st.subheader("Choosing the hybrid weights")
            st.dataframe(pd.read_csv(grid_file), hide_index=True)
    else:
        st.info("No results yet. Click **Run evaluation now** or run `python evaluate.py`.")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown("""
### Pipeline
1. **Data**: 63 real attractions in Jaipur and Delhi (tags, cost, duration, opening hours,
   location) and 300 simulated travellers with about 5,200 ratings.
2. **Content-based filtering**: each place becomes a TF-IDF vector of its tags. The
   traveller's profile is built from the interests they tick and/or the places they rated
   (weighted by rating minus their average). Score = cosine similarity.
3. **Item-based collaborative filtering**: places are similar if the same users rate them
   alike (adjusted cosine). A place's predicted rating is the traveller's mean plus the
   weighted deviations on its 10 most similar places they have rated.
4. **SVD (matrix factorisation)**: missing ratings are filled with the user's mean, the
   matrix is mean-centred, and only 8 latent factors are kept. Multiplying the factors back
   gives a predicted rating for every place.
5. **Hybrid**: *weighted* combination of the three (each scaled to 0–1). It *switches*
   to content-only for a new traveller (cold start).
6. **Itinerary planner (constraint-based)**: day by day, greedily pick the best place that
   is open on arrival, finishes before closing time and day end, fits the daily budget,
   and isn't too far (score − 0.02 × km). Stop when nothing fits and start the next day.
7. **Evaluation**: hide 20% of each user's ratings, retrain, compare RMSE/MAE and
   Precision/Recall/NDCG@5 against baselines.
""")
