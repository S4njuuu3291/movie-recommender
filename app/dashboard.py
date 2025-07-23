import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys, os
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import (
    recommend_for_user_table,
    recommend_surprise_for_user,
    recommend_ncf_for_user,
    recommend_cbf_for_user,
    recommend_switch_for_user_table,
    recommend_hybrid_for_user_table,
    recommend_list_for_user  # For popularity/random list if needed
)

# ---------- 1. Helper: Load Model/Data (Caching) ----------
@st.cache_resource
def load_models_and_data():
    with open("models/model_svd.pkl", "rb") as f:
        algo_svd = pickle.load(f)
    with open("models/model_usercf.pkl", "rb") as f:
        algo_user = pickle.load(f)
    with open("models/model_lightfm_popularity.pkl", "rb") as f:
        popularity_model = pickle.load(f)
    with open("models/model_lightfm_random.pkl", "rb") as f:
        random_model = pickle.load(f)
    model_ncf = tf.keras.models.load_model("models/model_ncf.h5", compile=False)
    with open("models/user_mapping.pkl", "rb") as f:
        user_mapping = pickle.load(f)
    with open("models/item_mapping.pkl", "rb") as f:
        item_mapping = pickle.load(f)
    tags_embedding_df = pd.read_pickle("models/tags_embedding.pkl")
    movies_df = pd.read_csv("data/raw/movies.csv")
    train_df = pd.read_csv("data/processed/train_df.csv")
    # For baseline: precomputed top-list (if available)
    try:
        with open("models/popularity_topk.pkl", "rb") as f:
            popularity_topk = pickle.load(f)
        with open("models/random_topk.pkl", "rb") as f:
            random_topk = pickle.load(f)
    except Exception:
        popularity_topk, random_topk = None, None
    return (algo_svd, algo_user, popularity_model, random_model, model_ncf,
            user_mapping, item_mapping, tags_embedding_df, movies_df, train_df,
            popularity_topk, random_topk)

# Load once (cache)
(algo_svd, algo_user, popularity_model, random_model, model_ncf,
 user_mapping, item_mapping, tags_embedding_df, movies_df, train_df,
 popularity_topk, random_topk) = load_models_and_data()

# ---------- 2. Sidebar: User Input ----------
st.sidebar.title("Recommendation Dashboard")
user_list = sorted(train_df['userId'].unique())
user_id = st.sidebar.selectbox("Pilih User", user_list)
top_k = st.sidebar.slider("Top K", 5, 20, 10)

model_option = st.sidebar.selectbox(
    "Pilih Model",
    ["Popularity", "Random", "UserCF", "SVD", "CBF", "Hybrid (Weighted)", "Hybrid (Switch)", "NCF"]
)

if model_option == "Hybrid (Weighted)":
    alpha = st.sidebar.slider("Alpha (SVD Weight)", 0.0, 1.0, 0.7)
if model_option == "Hybrid (Switch)":
    switch_threshold = st.sidebar.slider("Switch Threshold", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**About:**\n- MovieLens 100k\n- Baseline, CF, MF, CBF, Hybrid, NCF\n- Build by Sanjukin Pinem")

# ---------- 3. Main Panel: Recommend & Output ----------
st.title("🎬 Movie Recommendation System Dashboard")
st.write(f"**User ID:** {user_id}  |  **Model:** {model_option}  |  **Top {top_k}**")

if st.button("Get Recommendation"):
    if model_option == "Popularity":
        # Baseline: pakai LightFM popularity, mapping, dsb.
        result_df = recommend_for_user_table(
            user_id, popularity_model, user_mapping, item_mapping, train_df, movies_df, top_k
        )
        # Jika pakai list precomputed (faster), gunakan:
        # result_df = recommend_list_for_user(user_id, popularity_topk, train_df, movies_df, K=top_k)
    elif model_option == "Random":
        result_df = recommend_for_user_table(
            user_id, random_model, user_mapping, item_mapping, train_df, movies_df, top_k
        )
        # Jika pakai list precomputed (faster), gunakan:
        # result_df = recommend_list_for_user(user_id, random_topk, train_df, movies_df, K=top_k)
    elif model_option == "UserCF":
        result_df = recommend_surprise_for_user(user_id, algo_user, train_df, movies_df, K=top_k)
    elif model_option == "SVD":
        result_df = recommend_surprise_for_user(user_id, algo_svd, train_df, movies_df, K=top_k)
    elif model_option == "CBF":
        result_df = recommend_cbf_for_user(user_id, train_df, tags_embedding_df, movies_df, K=top_k)
    elif model_option == "Hybrid (Weighted)":
        # Load or compute svd_score, cbf_score
        with open("models/svd_score.pkl", "rb") as f:
            svd_score = pickle.load(f)
        with open("models/cbf_score.pkl", "rb") as f:
            cbf_score = pickle.load(f)
        result_df = recommend_hybrid_for_user_table(
            user_id, svd_score, cbf_score, alpha, movies_df, K=top_k
        )
    elif model_option == "Hybrid (Switch)":
        with open("models/recommendation_cbf.pkl", "rb") as f:
            recommendation_cbf = pickle.load(f)
        with open("models/recommendation_svd.pkl", "rb") as f:
            recommendation_svd = pickle.load(f)
        with open("models/user_train_count.pkl", "rb") as f:
            user_train_count = pickle.load(f)
        result_df = recommend_switch_for_user_table(
            user_id, train_df, recommendation_cbf, recommendation_svd,
            user_train_count, switch_threshold, movies_df, K=top_k
        )
    elif model_option == "NCF":
        result_df = recommend_ncf_for_user(
            user_id, model_ncf, user_mapping, item_mapping, train_df, movies_df, K=top_k
        )
    else:
        st.error("Model not implemented!")
        result_df = pd.DataFrame()
    if len(result_df) == 0:
        st.warning("Tidak ada rekomendasi ditemukan untuk user ini.")
    else:
        st.dataframe(result_df)
        # Optional: genre pie chart
        genre_list = result_df['genres'].str.split('|').explode()
        st.write("**Genre distribution in recommendation:**")
        st.bar_chart(genre_list.value_counts())

# ---------- 4. Profil & History User ----------
st.subheader("📑 User Profile & History")
user_history = train_df[train_df['userId'] == user_id]
history_df = user_history.merge(movies_df, on="movieId", how="left")[["title", "rating", "genres"]]
st.write("**Recently Rated Movies:**")
st.dataframe(history_df.sort_values("rating", ascending=False).head(10))

genre_profile = history_df['genres'].str.split('|').explode().value_counts().head(10)
st.write("**Top Genres Rated by User:**")
st.bar_chart(genre_profile)

# ---------- 5. Insight Section ----------
st.subheader("📝 Insight & Business Impact")
if model_option in ["Hybrid (Weighted)", "Hybrid (Switch)"]:
    st.success("Hybrid model menggabungkan keunggulan MF & CBF—lebih personal, dan tetap tahan cold start user!")
elif model_option == "CBF":
    st.info("CBF efektif untuk user baru, hasil lebih personal berdasar kemiripan konten/tag.")
elif model_option == "SVD":
    st.info("Matrix factorization cocok untuk user aktif—personalized, scalable.")
elif model_option == "UserCF":
    st.info("CF klasik relevan jika user banyak overlap history, tapi performa bisa turun jika data sangat sparse.")
elif model_option == "Popularity":
    st.warning("Model popularitas mudah, tapi tidak personal dan kurang baik untuk cold start.")
elif model_option == "Random":
    st.error("Random baseline—hanya untuk pembanding, bukan real recommendation.")
elif model_option == "NCF":
    st.info("NCF powerful jika data besar dan well-tuned. Potensi besar untuk ranking dan sequence-aware.")

st.markdown("""
---
**How it works?**  
- Semua model bisa dipilih via sidebar.
- Rekomendasi selalu personal dan bisa dijelaskan.
- Insight otomatis tiap model.
- Profil user & preferensi genre ditampilkan untuk analisis bisnis.
""")
