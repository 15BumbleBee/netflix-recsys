import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def average_precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    AP@K for a single user.
    recommended: ordered list of movie_ids (top-K)
    relevant: set of movie_ids the user actually rated >= 3.5
    """
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            precision_sum += hits / (i + 1)

    return precision_sum / min(len(relevant), k)


def compute_map_at_k(
    test_df: pd.DataFrame,
    model,
    train_df: pd.DataFrame,
    all_movie_ids: list,
    k: int = 10,
    relevance_threshold: float = 3.5,
    max_users: int = 500
) -> float:
    """
    Compute MAP@K across users.
    max_users: cap for speed during evaluation.
    """
    users = test_df["user_id"].unique()
    if len(users) > max_users:
        users = np.random.choice(users, size=max_users, replace=False)

    ap_scores = []
    for user_id in users:
        # Ground truth relevant items
        user_test = test_df[test_df["user_id"] == user_id]
        relevant = set(
            user_test[user_test["rating"] >= relevance_threshold]["movie_id"].tolist()
        )

        # Movies already rated by user in training
        rated = set(train_df[train_df["user_id"] == user_id]["movie_id"].tolist())

        # Get top-K recommendations
        recs = model.recommend_top_k(user_id, all_movie_ids, rated, k=k)
        recommended_ids = [r[0] for r in recs]

        ap = average_precision_at_k(recommended_ids, relevant, k=k)
        ap_scores.append(ap)

    return float(np.mean(ap_scores))


def evaluate_model(model_name: str, model, test_df: pd.DataFrame,
                   train_df: pd.DataFrame, all_movie_ids: list) -> dict:
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    # Rating predictions
    print("Computing RMSE and MAE...")
    y_pred = model.predict_batch(test_df)
    y_true = test_df["rating"].values

    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    # MAP@10
    print("Computing MAP@10 (this may take a moment)...")
    map10 = compute_map_at_k(
        test_df, model, train_df, all_movie_ids, k=10
    )
    print(f"MAP@10 : {map10:.4f}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "map@10": map10}