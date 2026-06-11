import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df: pd.DataFrame, movies: pd.DataFrame, save_dir: str = "outputs/eda"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 50)
    print("BASIC STATISTICS")
    print("=" * 50)
    print(f"Total ratings     : {len(df):,}")
    print(f"Unique users      : {df['user_id'].nunique():,}")
    print(f"Unique movies     : {df['movie_id'].nunique():,}")
    print(f"Rating range      : {df['rating'].min()} - {df['rating'].max()}")
    print(f"Mean rating       : {df['rating'].mean():.4f}")
    print(f"Median rating     : {df['rating'].median():.4f}")

    # Sparsity
    n_users = df["user_id"].nunique()
    n_movies = df["movie_id"].nunique()
    sparsity = 1 - len(df) / (n_users * n_movies)
    print(f"Matrix sparsity   : {sparsity * 100:.4f}%")

    # Ratings per user
    ratings_per_user = df.groupby("user_id")["rating"].count()
    print(f"\nRatings per user  : mean={ratings_per_user.mean():.1f}, "
          f"median={ratings_per_user.median():.1f}, "
          f"max={ratings_per_user.max()}")

    # Ratings per movie
    ratings_per_movie = df.groupby("movie_id")["rating"].count()
    print(f"Ratings per movie : mean={ratings_per_movie.mean():.1f}, "
          f"median={ratings_per_movie.median():.1f}, "
          f"max={ratings_per_movie.max()}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rating distribution
    sns.countplot(x="rating", data=df, ax=axes[0, 0], palette="Blues_d")
    axes[0, 0].set_title("Rating Distribution")
    axes[0, 0].set_xlabel("Rating")

    # 2. Ratings per user (log scale)
    axes[0, 1].hist(ratings_per_user, bins=50, color="steelblue", edgecolor="white")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Ratings per User (log scale)")
    axes[0, 1].set_xlabel("Number of Ratings")

    # 3. Ratings per movie (log scale)
    axes[1, 0].hist(ratings_per_movie, bins=50, color="coral", edgecolor="white")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Ratings per Movie (log scale)")
    axes[1, 0].set_xlabel("Number of Ratings")

    # 4. Top 20 most rated movies
    top_movies = ratings_per_movie.sort_values(ascending=False).head(20)
    top_movies_df = top_movies.reset_index()
    top_movies_df.columns = ["movie_id", "count"]
    top_movies_df = top_movies_df.merge(movies[["movie_id", "title"]], on="movie_id", how="left")

    axes[1, 1].barh(top_movies_df["title"].fillna(top_movies_df["movie_id"].astype(str)),
                    top_movies_df["count"], color="mediumseagreen")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Top 20 Most Rated Movies")
    axes[1, 1].set_xlabel("Number of Ratings")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_overview.png", dpi=150)
    plt.show()
    print(f"\nEDA plot saved to {save_dir}/eda_overview.png")

    return {
        "n_ratings": len(df),
        "n_users": n_users,
        "n_movies": n_movies,
        "sparsity": sparsity,
        "mean_rating": df["rating"].mean()
    }