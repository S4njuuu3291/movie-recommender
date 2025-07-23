import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- General recommend function for LightFM-like model (with mapping) ---
def recommend_for_user_table(user_id, model, user_mapping, item_mapping, train_df, movies, K=10):
    user_idx = user_mapping[user_id]
    rated_items = set(train_df[train_df['userId'] == user_id]['movieId'].unique())
    all_items = set(train_df['movieId'].unique())
    candidate_items = list(all_items - rated_items)
    candidate_indices = [item_mapping[mid] for mid in candidate_items if mid in item_mapping]
    if not candidate_indices:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    scores = model.predict(user_ids=np.full(len(candidate_indices), user_idx), item_ids=np.array(candidate_indices))
    top_k_idx = np.argsort(-scores)[:K]
    top_movieids = [candidate_items[i] for i in top_k_idx]
    df_result = pd.DataFrame({'movieId': top_movieids})
    df_result = df_result.merge(movies, on='movieId', how='left')
    return df_result[['movieId', 'title', 'genres']]

# --- Surprise CF/SVD model ---
def recommend_surprise_for_user(user_id, algo, train_df, movies, K=10):
    rated_items = set(train_df[train_df['userId'] == user_id]['movieId'].unique())
    all_items = set(train_df['movieId'].unique())
    candidate_items = list(all_items - rated_items)
    if not candidate_items:
        return pd.DataFrame(columns=['movieId', 'title', 'genres', 'score'])
    preds = [(iid, algo.predict(str(user_id), str(iid)).est) for iid in candidate_items]
    top_items = sorted(preds, key=lambda x: x[1], reverse=True)[:K]
    top_movieids = [iid for iid, score in top_items]
    top_scores = [score for iid, score in top_items]
    df_result = pd.DataFrame({'movieId': top_movieids, 'score': top_scores})
    df_result = df_result.merge(movies, on='movieId', how='left')
    return df_result[['movieId', 'title', 'genres', 'score']]

# --- NCF Model (Keras/Tensorflow) ---
def recommend_ncf_for_user(user_id, model, user2idx, item2idx, train_df, movies, K=10):
    rated_items = set(train_df[train_df['userId'] == user_id]['movieId'].unique())
    all_items = set(train_df['movieId'].unique())
    candidate_items = list(all_items - rated_items)
    if user_id not in user2idx or not candidate_items:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    u_idx = user2idx[user_id]
    candidate_idx = [item2idx[m] for m in candidate_items if m in item2idx]
    if not candidate_idx:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    user_arr = np.full(len(candidate_idx), u_idx)
    item_arr = np.array(candidate_idx)
    scores = model.predict([user_arr, item_arr], verbose=0).flatten()
    top_k_idx = np.argsort(-scores)[:K]
    top_movieids = [candidate_items[i] for i in top_k_idx]
    df_result = pd.DataFrame({'movieId': top_movieids})
    df_result = df_result.merge(movies, on='movieId', how='left')
    return df_result[['movieId', 'title', 'genres']]

# --- Content-Based Filtering (CBF) ---
def recommend_cbf_for_user(user_id, train_df, tags_embedding_df, movies, K=10):
    movie_ids_with_tags = set(tags_embedding_df.index)
    ratings = train_df[train_df['userId'] == user_id][['movieId', 'rating']]
    ratings = ratings[ratings['movieId'].isin(movie_ids_with_tags)]
    if len(ratings) == 0:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    movie_rated = ratings['movieId'].values
    unrated_movie_ids = list(movie_ids_with_tags - set(movie_rated))
    if not unrated_movie_ids:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    unrated = tags_embedding_df.loc[unrated_movie_ids]
    movie_vecs = tags_embedding_df.loc[movie_rated].values
    weights = ratings['rating'].values.reshape(-1, 1)
    user_profile = (movie_vecs * weights).sum(axis=0) / weights.sum()
    user_vec = user_profile.reshape(1, -1)
    sim_scores = cosine_similarity(user_vec, unrated.values)[0]
    top_n_idx = np.argpartition(-sim_scores, K)[:K]
    top_n_sorted = top_n_idx[np.argsort(-sim_scores[top_n_idx])]
    top_n_movieId = unrated.index[top_n_sorted]
    df_result = pd.DataFrame({'movieId': top_n_movieId})
    df_result = df_result.merge(movies, on='movieId', how='left')
    return df_result[['movieId', 'title', 'genres']]

# --- Hybrid Weighted (gabungan SVD & CBF score) ---
def recommend_hybrid_for_user_table(user_id, svd_score, cbf_score, alpha, movies, K=10):
    candidate_movies = set()
    candidate_movies.update([k[1] for k in svd_score if k[0] == user_id])
    candidate_movies.update([k[1] for k in cbf_score if k[0] == user_id])
    if not candidate_movies:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    scores = []
    for movie in candidate_movies:
        score = alpha * svd_score.get((user_id, movie), 0) + (1 - alpha) * cbf_score.get((user_id, movie), 0)
        scores.append((movie, score))
    top_movies = [movie for movie, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:K]]
    df_result = pd.DataFrame({'movieId': top_movies})
    df_result = df_result.merge(movies, on='movieId', how='left')
    return df_result[['movieId', 'title', 'genres']]

# --- Hybrid Switch ---
def recommend_switch_for_user_table(user_id, train_df, recommendation_cbf, recommendation_svd, user_train_count, switch_threshold, movies, K=10):
    if user_train_count.get(user_id, 0) < switch_threshold:
        rec_list = recommendation_cbf.get(user_id, [])[:K]
    else:
        rec_list = recommendation_svd.get(user_id, [])[:K]
    if not rec_list:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    df_result = pd.DataFrame({'movieId': rec_list})
    df_result = df_result.merge(movies, on='movieId', how='left')
    return df_result[['movieId', 'title', 'genres']]

# --- Evaluation Function ---
def evaluation(K, recommendation, test_data):
    precision_list, recall_list, hit_list = [], [], []
    for user, rec_items in recommendation.items():
        true_items = test_data[user]
        hits = set(true_items) & set(rec_items)
        precision = len(hits) / max(1, len(rec_items))
        recall = len(hits) / len(true_items)
        hit = 1 if len(hits) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        hit_list.append(hit)
    return {
        'precision@K_mean': np.mean(precision_list),
        'recall@K_mean': np.mean(recall_list),
        'hit@K_mean': np.mean(hit_list),
    }

# --- Baseline: Precomputed Recommendation List (Popularity/Random) ---
def recommend_list_for_user(user_id, rec_list, train_df, movies, K=10):
    rated_items = set(train_df[train_df['userId'] == user_id]['movieId'].unique())
    candidate_items = [m for m in rec_list if m not in rated_items]
    df_result = pd.DataFrame({'movieId': candidate_items[:K]})
    df_result = df_result.merge(movies, on='movieId', how='left')
    return df_result[['movieId', 'title', 'genres']]
