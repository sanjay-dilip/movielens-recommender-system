# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from pathlib import Path
# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# -------------------------
# Cached loaders
# -------------------------
@st.cache_data
def load_core_data():
    merged = pd.read_csv(PROCESSED_DIR / "merged.csv")
    movie_map = pd.read_csv(PROCESSED_DIR / "movie_map.csv")
    user_map = pd.read_csv(PROCESSED_DIR / "user_map.csv")
    pairs = pd.read_parquet(PROCESSED_DIR / "hybrid_train_pairs.parquet")
    return merged, movie_map, user_map, pairs
@st.cache_resource
def load_models_and_matrices():
    with open(PROCESSED_DIR / "als_model.pkl", "rb") as f:
        als_model = pickle.load(f)
    item_user = sp.load_npz(PROCESSED_DIR / "item_user_train.npz")
    user_item = item_user.T.tocsr()
    with open(PROCESSED_DIR / "hybrid_lgbm_model.pkl", "rb") as f:
        hybrid_model = pickle.load(f)
    return als_model, user_item, hybrid_model
# -------------------------
# Helper functions
# -------------------------
def get_user_history(merged, u_index, max_movies=15):
    user_rows = merged[merged["u_index"] == u_index].copy()
    if user_rows.empty:
        return pd.DataFrame(columns=["title", "genres", "rating", "timestamp"])
    user_rows = user_rows.sort_values("timestamp", ascending=False)
    hist = (
        user_rows[["title", "genres", "rating", "timestamp"]]
        .head(max_movies)
        .reset_index(drop=True)
    )
    return hist
def als_recommend_for_user(u_index, als_model, user_item_matrix, movie_map, top_n=10):
    if u_index < 0 or u_index >= als_model.user_factors.shape[0]:
        return pd.DataFrame(columns=["rank", "title", "genres", "als_score"])
    user_row = user_item_matrix[u_index]  # 1 x n_items CSR
    rec_ids, scores = als_model.recommend(
        userid=u_index,
        user_items=user_row,
        N=top_n,
    )
    if len(rec_ids) == 0:
        return pd.DataFrame(columns=["rank", "title", "genres", "als_score"])
    rec_df = movie_map[movie_map["m_index"].isin(rec_ids)].copy()
    order = {m: i for i, m in enumerate(rec_ids)}
    rec_df["rank"] = rec_df["m_index"].map(order)
    rec_df["als_score"] = rec_df["m_index"].map(
        {m: float(s) for m, s in zip(rec_ids, scores)}
    )
    rec_df = rec_df.sort_values("rank")
    return rec_df[["rank", "title", "genres", "als_score"]].reset_index(drop=True)
def hybrid_recommend_for_user(u_index, pairs_df, hybrid_model, feature_cols, movie_map, top_n=10):
    user_rows = pairs_df[pairs_df["u_index"] == u_index].copy()
    if user_rows.empty:
        return pd.DataFrame(columns=["title", "genres", "label", "score"])
    X_user = user_rows[feature_cols]   # keep as DataFrame so feature names match
    user_rows["score"] = hybrid_model.predict(X_user)
    user_rows = user_rows.sort_values("score", ascending=False)
    top = user_rows.head(top_n).merge(movie_map, on="m_index", how="left")
    return top[["title", "genres", "label", "score"]].reset_index(drop=True)
# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(
        page_title="Movie Recommender Demo",
        layout="wide",
    )
    st.title("🎬 Movie Recommender Demo")
    st.write(
        "This demo shows recommendations from two models: "
        "a collaborative ALS model and a hybrid model that combines ALS with content features."
    )
    # Load data and models
    merged, movie_map, user_map, pairs = load_core_data()
    als_model, user_item, hybrid_model = load_models_and_matrices()
    # Feature columns for hybrid model
    hybrid_feature_cols = [
        c for c in pairs.columns if c not in ["label", "u_index", "m_index"]
    ]
    # Sidebar controls
    st.sidebar.header("User selection")
    all_users = np.sort(merged["u_index"].unique())
    min_user, max_user = int(all_users.min()), int(all_users.max())
    selection_mode = st.sidebar.radio(
        "How to choose a user:",
        ["Random user", "Pick by index"],
    )
    if selection_mode == "Random user":
        u_index = int(np.random.choice(all_users))
    else:
        u_index = st.sidebar.number_input(
            "User index (internal u_index)",
            min_value=min_user,
            max_value=max_user,
            value=min_user,
            step=1,
        )
    top_n = st.sidebar.slider("Top N recommendations", 5, 30, 10)
    st.sidebar.write(f"Selected user index: `{u_index}`")
    # Main content
    st.subheader("User history")
    history_df = get_user_history(merged, u_index, max_movies=15)
    if history_df.empty:
        st.info("No history found for this user in the merged data.")
    else:
        st.dataframe(history_df)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ALS recommendations")
        als_df = als_recommend_for_user(u_index, als_model, user_item, movie_map, top_n=top_n)
        if als_df.empty:
            st.info("No ALS recommendations available for this user.")
        else:
            st.dataframe(als_df)
    with col2:
        st.subheader("Hybrid recommendations")
        hybrid_df = hybrid_recommend_for_user(
            u_index, pairs, hybrid_model, hybrid_feature_cols, movie_map, top_n=top_n
        )
        if hybrid_df.empty:
            st.info("No hybrid recommendations available for this user.")
        else:
            st.dataframe(hybrid_df)
    st.markdown("---")
    st.caption(
        "ALS uses collaborative filtering on the user–item matrix. "
        "The hybrid model adds genre features, user taste profiles, and ALS scores to a LightGBM ranker."
    )
if __name__ == "__main__":
    main()