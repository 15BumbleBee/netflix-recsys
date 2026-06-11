import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def load_netflix_data(data_dir: str, max_files: int = 4, sample_frac: float = 0.1):
    """
    Load Netflix Prize data from combined_data_*.txt files.
    sample_frac: fraction of users to keep (for memory management)
    """
    all_records = []
    
    for i in range(1, max_files + 1):
        filepath = os.path.join(data_dir, f"combined_data_{i}.txt")
        if not os.path.exists(filepath):
            print(f"File {filepath} not found, skipping.")
            continue
        
        print(f"Loading {filepath}...")
        current_movie_id = None
        
        with open(filepath, "r") as f:
            for line in tqdm(f, desc=f"File {i}"):
                line = line.strip()
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                else:
                    parts = line.split(",")
                    if len(parts) == 3:
                        user_id, rating, date = parts
                        all_records.append({
                            "movie_id": current_movie_id,
                            "user_id": int(user_id),
                            "rating": float(rating),
                            "date": date
                        })
    
    df = pd.DataFrame(all_records)
    print(f"Total records loaded: {len(df):,}")
    
    # Sample users for tractability
    if sample_frac < 1.0:
        unique_users = df["user_id"].unique()
        sampled_users = np.random.choice(
            unique_users,
            size=int(len(unique_users) * sample_frac),
            replace=False
        )
        df = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)
        print(f"After sampling {sample_frac*100:.0f}% users: {len(df):,} records")
    
    return df


def load_movie_titles(data_dir: str):
    """Load movie_titles.csv"""
    filepath = os.path.join(data_dir, "movie_titles.csv")
    movies = pd.read_csv(
        filepath,
        encoding="latin-1",
        header=None,
        names=["movie_id", "year", "title"],
        on_bad_lines="skip"
    )
    return movies