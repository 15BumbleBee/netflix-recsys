import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate


class SVDModel:
    """
    SVD via the Surprise library.
    Fast, well-tuned, strong baseline.
    """

    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr_all: float = 0.005, reg_all: float = 0.02):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.trainset = None

    def _to_surprise_dataset(self, df: pd.DataFrame):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
        return data

    def fit(self, train_df: pd.DataFrame):
        data = self._to_surprise_dataset(train_df)
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)
        print("SVD model trained.")

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        pred = self.model.predict(user_id, movie_id)
        return pred.est

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in test_df.iterrows():
            p = self.model.predict(row["user_id"], row["movie_id"])
            preds.append(p.est)
        return np.array(preds)

    def recommend_top_k(self, user_id: int, all_movie_ids: list,
                        rated_movie_ids: set, k: int = 10) -> list:
        """Return top-K (movie_id, predicted_rating) for unseen movies."""
        candidates = [m for m in all_movie_ids if m not in rated_movie_ids]
        scores = [(m, self.model.predict(user_id, m).est) for m in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]