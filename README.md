# MovieLens Recommender System – Pipeline, ALS, Hybrid Model, Streamlit App

## 📌 Overview

This project is an improved and complete version of my earlier [MovieLens](https://github.com/sanjay-dilip/Data-Science-Projects/tree/main/movielens) notebook that focused on EDA and a simple collaborative baseline.
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

## 🗂️ Data

Source: MovieLens (GroupLens) – standard **ratings.csv**, **movies.csv**

Schema used:

- `ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`

- `movies.csv`: `movieId`, `title`, `genres`

The pipeline converts raw IDs into numeric indices, cleans the data, and prepares all files in **data/processed/**.

---

## 🔧 Data Pipeline (src/pipeline)

The pipeline handles all preparation stages:

- Load raw ratings and movie files

- Clean missing rows

- Assign integer `u_index` and `m_index`

- Merge ratings with movie info

- Split train and test **per user** based on time

= Extract one row per movie per genre

- Build the **item-user sparse matrix** for ALS

- Save all processed files into `data/processed/`

This keeps everything reproducible and helps the notebooks stay clean.

---

## 🔎 Notebooks Breakdown

# 1. EDA

Explores the processed dataset:

- rating distribution

- user and movie activity

- genre counts

- basic behavior patterns

# 2. ALS Baseline Model

Uses the **implicit** library to train a collaborative filtering model:

- builds the user-item matrix

- trains ALS

- evaluates with Recall@K and NDCG@K

- saves the ALS model

This acts as the baseline recommender.

---

# 3. Content Feature Engineering

Builds all signals needed for a hybrid model:

- movie genre one hot vectors

- user taste profiles from past ratings

- user-movie genre similarity score

- ALS score for each user-movie pair

- positive and negative samples

- Outputs **hybrid_train_pairs.parquet**.

# 4. Hybrid LightGBM Ranker

Trains a ranking model that uses both ALS and content features.

- trains **LGBMRanker**

- uses group to rank movies per user

- evaluates with NDCG@K

- saves final hybrid model

This gives stronger recommendations than ALS alone.

# 5. Recommendation Notebook

Loads both trained models and shows:

- a user's movie history

- ALS-based recommendations

- hybrid recommendations with ranking scores

Useful for understanding how the system behaves.

## 🌐 Streamlit App

The Streamlit app provides a simple front end to explore the recommender.

It lets you:

- pick a user

- view their watch history

- see ALS recommendations

- see hybrid model recommendations

Run it using:

**streamlit run app.py**

## 🧩 Architecture Overview

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
