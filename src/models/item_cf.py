import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


class ItemCF:
    """
    Item-Based Collaborative Filtering using cosine similarity.
    """

    def __init__(self, k_neighbors: int = 20):
        self.k = k_neighbors
        self.sim_matrix = None
        self.user_item_matrix = None
        self.user2idx = None
        self.movie2idx = None
        self.idx2movie = None
        self.global_mean = 3.0

    def fit(self, train_df: pd.DataFrame, user2idx: dict, movie2idx: dict):
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.idx2movie = {i: m for m, i in movie2idx.items()}
        self.global_mean = train_df["rating"].mean()

        n_users = len(user2idx)
        n_movies = len(movie2idx)

        rows = train_df["user_id"].map(user2idx).values
        cols = train_df["movie_id"].map(movie2idx).values
        data = train_df["rating"].values

        self.user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_users, n_movies)
        )

        # Item similarity: item × user matrix
        item_user = self.user_item_matrix.T  # shape: (n_movies, n_users)
        print("Computing item-item cosine similarity...")
        self.sim_matrix = cosine_similarity(item_user, dense_output=False)
        print("Item-CF model ready.")

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        uidx = self.user2idx.get(user_id)
        midx = self.movie2idx.get(movie_id)
        if uidx is None or midx is None:
            return self.global_mean

        # Find movies rated by this user
        user_rated = self.user_item_matrix[uidx].nonzero()[1]
        if len(user_rated) == 0:
            return self.global_mean

        # Similarities between target movie and user-rated movies
        sims = np.array(self.sim_matrix[midx, user_rated].todense()).flatten()
        ratings = np.array(self.user_item_matrix[uidx, user_rated].todense()).flatten()

        # Top-k neighbors
        top_k_idx = np.argsort(sims)[-self.k:]
        top_k_sims = sims[top_k_idx]
        top_k_ratings = ratings[top_k_idx]

        denom = np.sum(np.abs(top_k_sims))
        if denom == 0:
            return self.global_mean

        return float(np.dot(top_k_sims, top_k_ratings) / denom)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in test_df.iterrows():
            preds.append(self.predict_rating(int(row["user_id"]), int(row["movie_id"])))
        return np.array(preds)

    def recommend_top_k(self, user_id: int, all_movie_ids: list,
                        rated_movie_ids: set, k: int = 10) -> list:
        candidates = [m for m in all_movie_ids if m not in rated_movie_ids]
        scores = [(m, self.predict_rating(user_id, m)) for m in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]