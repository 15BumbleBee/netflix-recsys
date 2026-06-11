import pandas as pd


def get_recommendations(model, user_id: int, train_df: pd.DataFrame,
                        movies_df: pd.DataFrame, all_movie_ids: list,
                        k: int = 10) -> pd.DataFrame:
    """
    Generate and display top-K recommendations for a user with movie metadata.
    """
    rated = set(train_df[train_df["user_id"] == user_id]["movie_id"].tolist())
    recs = model.recommend_top_k(user_id, all_movie_ids, rated, k=k)

    rec_df = pd.DataFrame(recs, columns=["movie_id", "predicted_score"])
    rec_df = rec_df.merge(movies_df[["movie_id", "title", "year"]], on="movie_id", how="left")
    rec_df["rank"] = range(1, len(rec_df) + 1)

    return rec_df[["rank", "title", "year", "predicted_score"]]


def explain_recommendation(user_id: int, movie_id: int,
                            train_df: pd.DataFrame, movies_df: pd.DataFrame,
                            item_cf_model) -> str:
    """
    Simple explanation: 'Users who liked X also liked Y.'
    Works with ItemCF similarity matrix.
    """
    midx = item_cf_model.movie2idx.get(movie_id)
    if midx is None:
        return "No explanation available."

    # Find top similar movies the user actually rated highly
    user_rated = train_df[
        (train_df["user_id"] == user_id) & (train_df["rating"] >= 4.0)
    ]
    if user_rated.empty:
        return "No explanation available (no high-rated history)."

    import numpy as np
    sim_scores = []
    for _, row in user_rated.iterrows():
        src_midx = item_cf_model.movie2idx.get(row["movie_id"])
        if src_midx is None:
            continue
        sim = item_cf_model.sim_matrix[midx, src_midx]
        try:
            sim_val = float(sim.toarray()[0][0])
        except Exception:
            sim_val = float(sim)
        title = movies_df[movies_df["movie_id"] == row["movie_id"]]["title"].values
        title = title[0] if len(title) > 0 else str(row["movie_id"])
        sim_scores.append((title, sim_val))

    sim_scores.sort(key=lambda x: x[1], reverse=True)
    top_similar = [t for t, s in sim_scores[:3] if s > 0]

    target_title = movies_df[movies_df["movie_id"] == movie_id]["title"].values
    target_title = target_title[0] if len(target_title) > 0 else str(movie_id)

    if top_similar:
        return (f"Recommended '{target_title}' because you rated these highly: "
                f"{', '.join(top_similar)}.")
    return f"Recommended '{target_title}' based on your viewing history."