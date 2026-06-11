import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_netflix_data, load_movie_titles
from src.eda import run_eda
from src.preprocessing import filter_data, encode_ids, split_data
from src.models.svd_model import SVDModel
from src.models.als_model import ALSModel
from src.models.item_cf import ItemCF
from src.evaluation import evaluate_model
from src.recommender import get_recommendations, explain_recommendation


DATA_DIR = "data/raw"
SAMPLE_FRAC = 0.05      # Use 5% of users — increase if you have more RAM
MIN_RATINGS = 20        # Min ratings per user and movie
TEST_SIZE = 0.2


def main():
    # -------------------------
    # 1. Load Data
    # -------------------------
    print("\n[1/6] Loading data...")
    df = load_netflix_data(DATA_DIR, max_files=1, sample_frac=SAMPLE_FRAC)
    movies = load_movie_titles(DATA_DIR)

    # -------------------------
    # 2. EDA
    # -------------------------
    print("\n[2/6] Running EDA...")
    run_eda(df, movies)

    # -------------------------
    # 3. Preprocessing
    # -------------------------
    print("\n[3/6] Preprocessing...")
    df = filter_data(df, min_user_ratings=MIN_RATINGS, min_movie_ratings=MIN_RATINGS)
    df, user2idx, movie2idx, idx2user, idx2movie = encode_ids(df)
    train_df, test_df = split_data(df, test_size=TEST_SIZE)

    all_movie_ids = list(df["movie_id"].unique())

    # -------------------------
    # 4. Train Models
    # -------------------------
    print("\n[4/6] Training models...")

    # SVD
    svd = SVDModel(n_factors=100, n_epochs=20)
    svd.fit(train_df)

    # ALS
    als = ALSModel(factors=100, iterations=20)
    als.fit(train_df, user2idx, movie2idx)

    # Item-CF
    icf = ItemCF(k_neighbors=20)
    icf.fit(train_df, user2idx, movie2idx)

    # -------------------------
    # 5. Evaluate
    # -------------------------
    print("\n[5/6] Evaluating models...")
    results = []
    for name, model in [("SVD", svd), ("ALS", als), ("ItemCF", icf)]:
        res = evaluate_model(name, model, test_df, train_df, all_movie_ids)
        results.append(res)

    results_df = pd.DataFrame(results)
    print("\n\nFINAL RESULTS COMPARISON")
    print(results_df.to_string(index=False))

    # -------------------------
    # 6. Recommendations
    # -------------------------
    print("\n[6/6] Generating sample recommendations...")
    sample_users = train_df["user_id"].unique()[:3]

    for uid in sample_users:
        print(f"\n--- Top-10 Recommendations for User {uid} (SVD) ---")
        recs = get_recommendations(svd, uid, train_df, movies, all_movie_ids, k=10)
        print(recs.to_string(index=False))

        # Explanation (ItemCF)
        if len(recs) > 0:
            first_movie_id = movies[movies["title"] == recs.iloc[0]["title"]]["movie_id"].values
            if len(first_movie_id) > 0:
                explanation = explain_recommendation(
                    uid, first_movie_id[0], train_df, movies, icf
                )
                print(f"\nExplanation: {explanation}")


if __name__ == "__main__":
    main()