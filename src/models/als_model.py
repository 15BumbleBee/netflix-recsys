import numpy as np
import pandas as pd
import scipy.sparse as sp
import implicit


class ALSModel:
    """
    Implicit ALS (implicit feedback via confidence weighting).
    We treat ratings >= 3.5 as positive implicit signals.
    """

    def __init__(self, factors: int = 100, iterations: int = 20,
                 regularization: float = 0.1, alpha: float = 40.0):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=42
        )
        self.alpha = alpha
        self.user2idx = None
        self.movie2idx = None
        self.idx2movie = None
        self.user_item_matrix = None
        self.global_mean = 3.5
        self.n_users = 0
        self.n_movies = 0

    def fit(self, train_df: pd.DataFrame, user2idx: dict, movie2idx: dict):
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.idx2movie = {i: m for m, i in movie2idx.items()}
        self.global_mean = train_df["rating"].mean()

        self.n_users = len(user2idx)
        self.n_movies = len(movie2idx)

        rows = train_df["user_id"].map(user2idx).values
        cols = train_df["movie_id"].map(movie2idx).values
        data = 1 + self.alpha * train_df["rating"].values

        self.user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.n_users, self.n_movies)
        )

        item_user = self.user_item_matrix.T.tocsr()
        self.model.fit(item_user)

        self.trained_n_users = self.model.user_factors.shape[0]
        self.trained_n_items = self.model.item_factors.shape[0]

        print(f"ALS model trained. user_factors: {self.model.user_factors.shape}, "
              f"item_factors: {self.model.item_factors.shape}")

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        uidx = self.user2idx.get(user_id)
        midx = self.movie2idx.get(movie_id)

        if uidx is None or midx is None:
            return self.global_mean

        if uidx >= self.trained_n_users or midx >= self.trained_n_items:
            return self.global_mean

        score = self.model.user_factors[uidx] @ self.model.item_factors[midx]
        return float(np.clip(score, 1, 5))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in test_df.iterrows():
            preds.append(self.predict_rating(int(row["user_id"]), int(row["movie_id"])))
        return np.array(preds)

    def recommend_top_k(self, user_id: int, all_movie_ids: list,
                        rated_movie_ids: set, k: int = 10) -> list:
        uidx = self.user2idx.get(user_id)
        if uidx is None:
            return []

        if uidx >= self.trained_n_users:
            return []

        # Manual approach: score all items using dot product, then rank
        user_vec = self.model.user_factors[uidx]  # shape: (factors,)
        scores = self.model.item_factors @ user_vec  # shape: (n_items,)

        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for item_idx in ranked_indices:
            movie_id = self.idx2movie.get(int(item_idx))
            if movie_id is None:
                continue
            if movie_id in rated_movie_ids:
                continue
            results.append((movie_id, float(scores[item_idx])))
            if len(results) == k:
                break

        return results