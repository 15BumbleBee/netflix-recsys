import pandas as pd
import numpy as np
from src.preprocessing import filter_data, encode_ids, split_data
from src.models.svd_model import SVDModel
from src.evaluation import compute_rmse, compute_mae

# Synthetic data
np.random.seed(42)
df = pd.DataFrame({
    "user_id": np.random.randint(1, 51, 2000),
    "movie_id": np.random.randint(1, 101, 2000),
    "rating": np.random.choice([1,2,3,4,5], 2000),
    "date": "2023-01-01"
})
df = df.drop_duplicates(["user_id","movie_id"])

df = filter_data(df, min_user_ratings=5, min_movie_ratings=5)
df, u2i, m2i, i2u, i2m = encode_ids(df)
train, test = split_data(df)

model = SVDModel(n_factors=20, n_epochs=5)
model.fit(train)

preds = model.predict_batch(test)
print(f"RMSE: {compute_rmse(test['rating'].values, preds):.4f}")
print(f"MAE : {compute_mae(test['rating'].values, preds):.4f}")
print("Sanity check passed!")