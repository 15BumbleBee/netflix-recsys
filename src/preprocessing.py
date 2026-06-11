import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def filter_data(df: pd.DataFrame, min_user_ratings: int = 20, min_movie_ratings: int = 20):
    """Remove sparse users and movies."""
    print(f"Before filtering: {len(df):,} ratings")
    
    user_counts = df["user_id"].value_counts()
    movie_counts = df["movie_id"].value_counts()
    
    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    
    df = df[df["user_id"].isin(valid_users) & df["movie_id"].isin(valid_movies)]
    print(f"After filtering  : {len(df):,} ratings | "
          f"{df['user_id'].nunique():,} users | {df['movie_id'].nunique():,} movies")
    return df.reset_index(drop=True)


def encode_ids(df: pd.DataFrame):
    """Map user/movie IDs to contiguous 0-indexed integers."""
    user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
    movie2idx = {m: i for i, m in enumerate(df["movie_id"].unique())}
    
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["movie_idx"] = df["movie_id"].map(movie2idx)
    
    idx2user = {i: u for u, i in user2idx.items()}
    idx2movie = {i: m for m, i in movie2idx.items()}
    
    return df, user2idx, movie2idx, idx2user, idx2movie


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified split by user to ensure every user appears in both train/test."""
    train_list, test_list = [], []
    
    for _, group in df.groupby("user_id"):
        if len(group) < 2:
            train_list.append(group)
            continue
        tr, te = train_test_split(group, test_size=test_size, random_state=random_state)
        train_list.append(tr)
        test_list.append(te)
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df