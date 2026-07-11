# MovieLens Recommender System

## Overview

This project is an improved and complete version of an earlier [MovieLens data analysis](https://github.com/sanjay-dilip/movielens-data-analysis) notebook that focused on EDA and a simple collaborative baseline.
Here, the goal is to build a **full movie recommender system** that helps users find movies that match their taste.

People face too many choices and basic popularity lists do not help.
A good recommender should learn from:

- what a user watched or rated
- what similar users liked
- what a movie contains

This system brings all these ideas together.

The project uses the MovieLens dataset and includes:

- a data pipeline
- an ALS collaborative model
- a hybrid LightGBM ranking model
- a Streamlit app

---

## Dataset

Source: MovieLens (GroupLens) – raw files **ratings.dat**, **movies.dat** (`::`-delimited)

Schema used:

- `ratings.dat`: `user_id`, `movie_id`, `rating`, `timestamp`
- `movies.dat`: `movie_id`, `title`, `genres`

The pipeline converts raw IDs into numeric indices, cleans the data, and prepares all files in `data/processed/`.

---

## System Architecture

```
                          Raw MovieLens Data
                                  |
                                  v
                         Data Pipeline (src/)
                                  |
                 -------------------------------------
                 |                                   |
                 v                                   v
        ALS Collaborative Model            Content Feature Builder
        - sparse matrix                    - movie genres
        - user factors                     - user profiles
        - item factors                     - similarity
                                           - ALS score
                 |                                   |
                 ---------------   -------------------
                                 v
                        Hybrid Ranking Model
                            LightGBM
                                 |
                                 v
                          Streamlit App
                        - user history
                        - ALS results
                        - hybrid results
```

## Data Preparation Pipeline (`src/pipeline`)

The pipeline handles all preparation stages:

- Load raw ratings and movie files
- Clean missing rows
- Assign integer `u_index` and `m_index`
- Merge ratings with movie info
- Split train and test **per user** based on time
- Extract one row per movie per genre
- Build the **item-user sparse matrix** for ALS
- Save all processed files into `data/processed/`

Run it with:
```bash
python src/run_pipeline.py
```

This keeps everything reproducible and helps the notebooks stay clean.

---

## ALS Collaborative Filtering (`notebooks/als.ipynb`)

Uses the **implicit** library to train a collaborative filtering model:

- builds the user-item matrix
- trains ALS (`factors=64`, `regularization=0.1`, `iterations=20`)
- evaluates with Recall@10 and NDCG@10
- saves the ALS model to `data/processed/als_model.pkl`

This acts as the baseline recommender.

## Hybrid LightGBM Ranking (`notebooks/hybrid_ranker.ipynb`)

Trains a ranking model that uses both ALS and content features (`notebooks/content_features.ipynb` builds the inputs):

- movie genre one-hot vectors, user taste profiles, user-movie genre similarity, ALS score per user-movie pair, positive/negative samples (output: `hybrid_train_pairs.parquet`)
- trains **LGBMRanker** with `lambdarank` objective, grouped per user
- evaluates with NDCG@K
- saves the final model to `data/processed/hybrid_lgbm_model.pkl`

## Evaluation Workflow

The original per-model evaluation ran inline within the training notebooks (`als.ipynb`, `hybrid_ranker.ipynb`). A shared, unified evaluation module (`src/evaluation.py`, run via `python src/run_evaluation.py`) now scores both **frozen, already-trained** models — no retraining — against identical per-user candidate groups built from the held-out time-based test split, so the two models' numbers are directly comparable.

### Current Verified Metrics

**Legacy per-notebook snapshots (different protocols, not directly comparable — kept for history):**

| Model | Metric | Value |
|---|---|---|
| ALS (factors=64, reg=0.1, iters=20) | Recall@10 | 0.0017 |
| ALS (factors=64, reg=0.1, iters=20) | NDCG@10 | 0.0057 |
| Hybrid LightGBM Ranker | Average NDCG | 0.9585 |

These aren't just "different protocols" — the ALS notebook's evaluation helper silently dropped ~38.6% of users due to an indexing bound checked against the wrong matrix shape, and the hybrid ranker's NDCG was computed against **train-derived labels** over a candidate pool of train positives + negatives, which measures training-set re-ranking rather than generalization to held-out data.

**Unified evaluation protocol:** both models scored against the same held-out `test.csv` positives, the same per-user candidate groups (all held-out positives + 99 sampled true negatives, excluded from both train and test interactions), and the same @10 cutoff, over 3 negative-sampling seeds (42, 43, 44 — models are not retrained, only the negative sampling varies across seeds):

| Model | Recall@10 (mean±std) | NDCG@10 (mean±std) | Precision@10 (mean±std) |
|---|---|---|---|
| ALS (factors=64, reg=0.1, iters=20) | 0.4107 ± 0.0006 | 0.7243 ± 0.0017 | 0.6467 ± 0.0003 |
| Hybrid LightGBM Ranker | 0.2269 ± 0.0008 | 0.4734 ± 0.0014 | 0.4278 ± 0.0014 |

**Note on comparability:** under this corrected, held-out protocol, ALS actually outperforms the hybrid ranker — the reverse of what the legacy snapshot suggested, since the legacy hybrid NDCG measured re-ranking of training labels rather than generalization. This is a measurement correction, not a model regression, and it's a more honest starting point for the refinement work below. The unified numbers are also considerably higher than the legacy ALS Recall@10/NDCG@10 because ranking against ~100 candidates (positives + 99 sampled negatives) is a substantially easier task than ranking against the full ~3,706-movie catalog — both are legitimate, standard evaluation setups, but they aren't comparable to each other either.

### Future Evaluation & Metric Refinement
This repository is the active target for further work. Planned/pending items:
- [x] Re-evaluate ALS and the hybrid ranker under one consistent protocol (same candidate set, same @K, same test split).
- [ ] Add Precision@K and coverage/diversity metrics alongside Recall/NDCG.
- [ ] Cross-validate metrics across multiple random seeds / splits to check stability.
- [ ] Produce resume-ready, defensible summary metrics once the above is complete.

Until the remaining items land, treat the unified numbers above as a solid but not final benchmark.

## Streamlit Application

The Streamlit app (`app.py`) provides a simple front end to explore the recommender.

It lets you:

- pick a user
- view their watch history
- see ALS recommendations
- see hybrid model recommendations

Run it using:
```bash
streamlit run app.py
```

Live deployment: [Streamlit App](https://movielens-movie-recommender.streamlit.app/)

## Repository Structure
```
movielens-recommender-system/
├── app.py
├── config.yaml
├── requirements.txt
├── data/
│   ├── movies_ratings_cleaned.csv
│   ├── raw/
│   │   ├── movies.dat
│   │   └── ratings.dat
│   └── processed/
│       ├── merged.csv, train.csv, test.csv
│       ├── movie_map.csv, user_map.csv, movie_genres.csv
│       ├── item_user_train.npz
│       ├── hybrid_train_pairs.parquet
│       ├── als_model.pkl, hybrid_lgbm_model.pkl
│       └── eda_summary.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── als.ipynb
│   ├── content_features.ipynb
│   ├── hybrid_ranker.ipynb
│   └── recommendations.ipynb
├── src/
│   ├── run_pipeline.py
│   ├── run_evaluation.py
│   ├── evaluation.py
│   └── pipeline/
│       ├── io.py, processing.py, splits.py, matrices.py
└── README.md
```

## Setup & Execution
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Rebuild processed data from raw (optional — data/processed/ is already included)
python src/run_pipeline.py

# Re-run the unified ALS vs. hybrid evaluation (scores existing trained models, no retraining)
python src/run_evaluation.py

# Run the Streamlit app
streamlit run app.py
```
All paths in `app.py`, `src/run_pipeline.py`, and the notebooks are resolved relative to the project's own root (`Path(__file__).resolve().parent` / `Path("..").resolve()`), so the project runs standalone with no dependency on the old monorepo layout.

## Related Project
This project was inspired by an earlier exploratory analysis of the MovieLens dataset, but it is maintained as an independent end-to-end recommendation system: [movielens-data-analysis](https://github.com/sanjay-dilip/movielens-data-analysis).
