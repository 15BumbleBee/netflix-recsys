# Recommendation Systems for Personalized Content Discovery

> Built on the Netflix Prize Dataset as part of a Machine Learning Competition — 2025

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Models](https://img.shields.io/badge/Models-SVD%20%7C%20ItemCF%20%7C%20ALS-red)
![Dataset](https://img.shields.io/badge/Dataset-Netflix%20Prize-black?logo=netflix)

---

## What This Project Does

Every time you open Netflix and it somehow knows exactly what you want to watch — that's a recommendation system at work. This project builds one from scratch using the Netflix Prize Dataset, one of the most well-known benchmarks in machine learning.

We trained and compared three different approaches to the same problem: given a user's past ratings, predict what they'll enjoy next. The results were clear — matrix factorization (SVD) consistently outperformed the other two methods on both rating accuracy and ranking quality.

| Model | Type | RMSE | MAE | MAP@10 |
|-------|------|------|-----|--------|
| **SVD** | Matrix Factorization | **0.9044** | **0.7049** | **0.0140** |
| Item-CF | Item-Based Collaborative Filtering | 0.9588 | 0.7479 | 0.0090 |
| ALS | Implicit Alternating Least Squares | 1.8174 | 1.4461 | 0.0005 |

---

## Project Structure

```
netflix-recsys/
├── data/
│   └── raw/                  # Place Netflix Prize files here (not tracked by git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Parses Netflix Prize raw files
│   ├── eda.py                # Exploratory Data Analysis + plots
│   ├── preprocessing.py      # Filtering, encoding, train/test split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── svd_model.py      # SVD via Surprise library
│   │   ├── als_model.py      # ALS via implicit library
│   │   └── item_cf.py        # Item-Based CF via cosine similarity
│   ├── evaluation.py         # RMSE, MAE, MAP@10 computation
│   └── recommender.py        # Top-K generation + explanation
├── notebooks/
│   └── analysis.ipynb        # EDA notebook
├── main.py                   # Full pipeline entry point
└── requirements.txt
```

---

## Dataset

We used the **Netflix Prize Dataset**, available on [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).

| Metric | Value |
|--------|-------|
| Total Ratings | 100,480,507 |
| Users | 480,189 |
| Movies | 17,770 |
| Rating Scale | 1 to 5 stars |
| Matrix Sparsity | ~98.9% |

The dataset is massive and not included in this repository. After downloading, place the files like this:

```
data/raw/
├── combined_data_1.txt
├── combined_data_2.txt   # optional
├── combined_data_3.txt   # optional
├── combined_data_4.txt   # optional
└── movie_titles.csv
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/15BumbleBee/netflix-recsys.git
cd netflix-recsys
```

**2. Create a virtual environment**

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Head over to the [Kaggle page](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data), download the files, and place them in `data/raw/` as shown above.

---

## Running the Pipeline

```bash
python main.py
```

The script runs all six steps back to back:

```
[1/6] Loading data
[2/6] Exploratory Data Analysis
[3/6] Preprocessing
[4/6] Training models (SVD, ALS, ItemCF)
[5/6] Evaluating models (RMSE, MAE, MAP@10)
[6/6] Generating sample recommendations
```

If your machine struggles with memory, you can reduce the load by editing these lines at the top of `main.py`:

```python
SAMPLE_FRAC = 0.05    # percentage of users to sample
MIN_RATINGS = 20      # minimum ratings required per user/movie
TEST_SIZE   = 0.2     # 80/20 train-test split
```

---

## Quick Test (no dataset needed)

To verify the pipeline works before downloading the full dataset:

```bash
python test_quick.py
```

This generates synthetic rating data and runs the SVD model end-to-end. If it prints an RMSE and "Sanity check passed", everything is working.

---

## Sample Recommendations

These are real outputs from our SVD model.

**User 250** — seems to enjoy action and sci-fi (rated The Mummy, Jaws, Pirates of the Caribbean highly)

| Rank | Title | Year | Predicted Score |
|------|-------|------|-----------------|
| 1 | Futurama: Monster Robot Maniac Fun Collection | 1999 | 4.54 |
| 2 | Stargate SG-1: Season 8 | 2004 | 4.52 |
| 3 | Star Trek: Voyager: Season 5 | 1998 | 4.50 |
| 4 | The Sixth Sense | 1999 | 4.44 |
| 5 | Lost: Season 1 | 2004 | 4.43 |

**User 1086** — comedy and classics (rated Clerks, American Beauty, Ace Ventura highly)

| Rank | Title | Year | Predicted Score |
|------|-------|------|-----------------|
| 1 | The Simpsons: Season 3 | 1991 | 4.80 |
| 2 | Yojimbo | 1961 | 4.61 |
| 3 | The Simpsons: Treehouse of Horror | 1990 | 4.57 |
| 4 | Family Guy: Freakin' Sweet Collection | 2004 | 4.48 |
| 5 | Curb Your Enthusiasm: Season 3 | 2002 | 4.48 |

The recommendations are noticeably different for each user, which is exactly what you want — the model has clearly picked up on individual taste rather than just recommending popular titles to everyone.

---

## Key Findings

**SVD wins on both metrics.** It achieves the lowest RMSE (0.90) and the highest MAP@10 (0.014), confirming that matrix factorization is the right approach for sparse, explicit rating data.

**ALS was the wrong tool for this job.** ALS was designed for implicit feedback like clicks and watch history — not star ratings. Using it here produces poor results, which is expected once you understand what it was built for.

**RMSE and MAP@10 measure different things.** Item-CF and SVD have fairly similar RMSE values, but SVD's MAP@10 is 56% higher. This shows that being accurate at predicting individual ratings does not automatically mean you're good at ranking recommendations — the two metrics capture different aspects of quality.

**The model learns taste, not just popularity.** Serialised TV content clusters naturally — users who rated one season of a show consistently receive recommendations for other seasons. That's a genuinely useful property for a streaming platform.

---

## Tech Stack

- Python 3.10+
- [scikit-surprise](https://surpriselib.com/) — SVD
- [implicit](https://github.com/benfred/implicit) — ALS
- [scikit-learn](https://scikit-learn.org/) — cosine similarity for Item-CF
- [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/) — data processing
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) — visualizations

---

## What We Would Do Next

- [ ] Neural Collaborative Filtering (NCF) with PyTorch — replace the dot product with MLP layers
- [ ] TimeSVD++ — use rating timestamps to model how taste changes over time
- [ ] Hybrid model — bring in genre and cast metadata to handle new users
- [ ] Train on the full dataset with GPU acceleration (expecting RMSE below 0.87)
- [ ] Deploy as a FastAPI service for real-time recommendations
- [ ] Post-processing for diversity — avoid recommending ten seasons of the same show

---

## Deliverables

| File | Description |
|------|-------------|
| `Technical_Report.pdf` | 10-page report covering EDA, methodology, results, and insights |
| `Presentation.pdf` | 8-slide deck for the competition presentation |

---

## Team

| Name | Contribution |
|------|-------------|
| [Anuj Jangir](https://github.com/15BumbleBee) | Data pipeline, SVD model, evaluation, report |
| [Rakshit Raj](https://github.com/rakshit11147) | ALS model, Item-CF, EDA, presentation |

---

## License

This project is for educational and competition purposes.
The Netflix Prize Dataset is subject to its own [terms of use](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
