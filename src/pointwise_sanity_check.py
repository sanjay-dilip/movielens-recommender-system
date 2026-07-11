"""Sanity check: does a pointwise LGBMClassifier (no lambdarank, no groups)
on the same features do better/worse than the ratio-fixed LGBMRanker?

Trains on the current (leakage-fixed, ratio-fixed) data/processed/hybrid_train_pairs.parquet
as a plain binary classification problem, then scores it through the existing
src/evaluation.py grouped protocol using predicted probability as the ranking score.

Does not modify hybrid_ranker.ipynb, the existing hybrid_lgbm_model.pkl, or
src/evaluation.py, and does not persist a new model artifact — this is a
one-off diagnostic run, not part of the maintained pipeline.
"""

import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.evaluation import (
    EVAL_SEEDS,
    FEATURE_COLUMN_ORDER,
    K,
    N_NEGATIVES,
    PROCESSED_DIR,
    build_candidate_groups,
    build_feature_matrix,
    build_interaction_sets,
    build_movie_genre_matrix,
    build_user_genre_profile,
    evaluate_model_for_seed,
    load_artifacts,
    verify_als_score_convention,
    verify_factor_shapes,
    verify_feature_column_order,
)


def train_pointwise_classifier(processed_dir: Path = PROCESSED_DIR) -> LGBMClassifier:
    pairs = pd.read_parquet(processed_dir / "hybrid_train_pairs.parquet")
    X = pairs[FEATURE_COLUMN_ORDER].to_numpy()
    y = pairs["label"].to_numpy()
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
    clf.fit(X, y)
    return clf


def rank_pointwise_group(
    clf: LGBMClassifier,
    u_index: int,
    candidate_m_indices: Sequence[int],
    user_genre_profile: pd.DataFrame,
    movie_genre_matrix: pd.DataFrame,
    als_model,
) -> List[int]:
    X = build_feature_matrix(u_index, candidate_m_indices, user_genre_profile, movie_genre_matrix, als_model)
    scores = clf.predict_proba(X)[:, 1]
    order = sorted(range(len(candidate_m_indices)), key=lambda i: (-scores[i], candidate_m_indices[i]))
    return [candidate_m_indices[i] for i in order]


def main():
    print("Verifying frozen-artifact assumptions...")
    artifacts = load_artifacts()
    verify_factor_shapes(artifacts)
    verify_feature_column_order()
    verify_als_score_convention(artifacts.als_model)

    print("Training pointwise LGBMClassifier on hybrid_train_pairs.parquet...")
    clf = train_pointwise_classifier()

    train_items_by_user, test_items_by_user = build_interaction_sets(artifacts.train, artifacts.test)
    movie_genre_matrix = build_movie_genre_matrix(artifacts.movie_genres, artifacts.movie_map)
    user_genre_profile = build_user_genre_profile(artifacts.train, movie_genre_matrix)
    all_movie_ids = artifacts.movie_map["m_index"].to_numpy()

    results = []
    for seed in EVAL_SEEDS:
        groups = build_candidate_groups(train_items_by_user, test_items_by_user, all_movie_ids, N_NEGATIVES, seed)
        metrics = evaluate_model_for_seed(
            lambda u, g: rank_pointwise_group(clf, u, g, user_genre_profile, movie_genre_matrix, artifacts.als_model),
            groups,
            test_items_by_user,
            K,
        )
        results.append({"seed": seed, **metrics})

    raw_df = pd.DataFrame(results)
    print("\nPer-seed results:\n", raw_df)

    summary = raw_df[["recall_at_10", "ndcg_at_10", "precision_at_10"]].agg(["mean", "std"])
    print("\nSummary (mean, std over seeds):\n", summary)

    print("\nMarkdown row for report:\n")
    r = summary["recall_at_10"]
    n = summary["ndcg_at_10"]
    p = summary["precision_at_10"]
    print(
        f"| Pointwise LGBMClassifier | {r['mean']:.4f} ± {r['std']:.4f} | "
        f"{n['mean']:.4f} ± {n['std']:.4f} | {p['mean']:.4f} ± {p['std']:.4f} |"
    )


if __name__ == "__main__":
    main()
