"""Unified evaluation of the ALS and hybrid LightGBM recommenders.

Both models are frozen (loaded from data/processed/*.pkl, never retrained
here) and scored against identical per-user candidate groups built from the
held-out test split, so their metrics are directly comparable.

Two orientation quirks in the existing pipeline matter here:

1. `als_model` was fit directly on item_user_train.npz (items x users)
   without transposing, so `als_model.user_factors` (3706 rows) is actually
   indexed by movie (m_index), and `als_model.item_factors` (6040 rows) is
   actually indexed by user (u_index). Direct ALS scoring in this module
   uses that corrected correspondence.
2. The frozen hybrid_lgbm_model.pkl was trained on an `als_score` feature
   computed with the *inverted* convention (dot(user_factors[u],
   item_factors[m])), which also zeroes out whenever u_index >= 3706 (an
   out-of-bounds guard written against the wrong shape). Since the model is
   frozen and has no feature-name validation, this exact convention (bug
   included) must be reproduced when scoring new candidate pairs, or the
   features are out-of-distribution for the model.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Sequence, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

K = 10
N_NEGATIVES = 99
EVAL_SEEDS = [42, 43, 44]

GENRE_ORDER = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
USER_GENRE_COLS = [f"genre_{g}_user" for g in GENRE_ORDER]
MOVIE_GENRE_COLS = [f"genre_{g}_movie" for g in GENRE_ORDER]
FEATURE_COLUMN_ORDER = USER_GENRE_COLS + MOVIE_GENRE_COLS + ["genre_sim", "als_score"]


@dataclass
class Artifacts:
    train: pd.DataFrame
    test: pd.DataFrame
    user_map: pd.DataFrame
    movie_map: pd.DataFrame
    movie_genres: pd.DataFrame
    als_model: object
    hybrid_model: object


def load_artifacts(processed_dir: Path = PROCESSED_DIR) -> Artifacts:
    """Read the existing pipeline/model outputs. No writes, no retraining."""
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    user_map = pd.read_csv(processed_dir / "user_map.csv")
    movie_map = pd.read_csv(processed_dir / "movie_map.csv")
    movie_genres = pd.read_csv(processed_dir / "movie_genres.csv")
    with open(processed_dir / "als_model.pkl", "rb") as f:
        als_model = pickle.load(f)
    with open(processed_dir / "hybrid_lgbm_model.pkl", "rb") as f:
        hybrid_model = pickle.load(f)
    return Artifacts(train, test, user_map, movie_map, movie_genres, als_model, hybrid_model)


def verify_factor_shapes(artifacts: Artifacts) -> None:
    """Fail loudly if the ALS orientation assumption this module relies on ever breaks."""
    n_users = len(artifacts.user_map)
    n_movies = len(artifacts.movie_map)
    got_item_factors = artifacts.als_model.item_factors.shape[0]
    got_user_factors = artifacts.als_model.user_factors.shape[0]
    assert got_item_factors == n_users, (
        f"expected als_model.item_factors to have {n_users} rows (one per user), "
        f"got {got_item_factors} — the ALS orientation assumption this module "
        f"relies on no longer holds."
    )
    assert got_user_factors == n_movies, (
        f"expected als_model.user_factors to have {n_movies} rows (one per movie), "
        f"got {got_user_factors} — the ALS orientation assumption this module "
        f"relies on no longer holds."
    )


def verify_feature_column_order(processed_dir: Path = PROCESSED_DIR) -> None:
    """Re-derive feature_cols live from the training parquet exactly as
    hybrid_ranker.ipynb does, and assert it matches FEATURE_COLUMN_ORDER.
    The frozen LGBMRanker has no feature-name validation, so a silent
    reorder would corrupt every prediction without raising an error."""
    pairs = pd.read_parquet(processed_dir / "hybrid_train_pairs.parquet")
    feature_cols = [c for c in pairs.columns if c not in ("label", "u_index", "m_index")]
    assert feature_cols == FEATURE_COLUMN_ORDER, (
        "hybrid_train_pairs.parquet's feature column order no longer matches "
        "FEATURE_COLUMN_ORDER — update the constant to match."
    )


def _als_score_training_convention(als_model, u_index: int, m_index: int) -> float:
    if u_index >= als_model.user_factors.shape[0]:
        return 0.0
    if m_index >= als_model.item_factors.shape[0]:
        return 0.0
    return float(np.dot(als_model.user_factors[u_index], als_model.item_factors[m_index]))


def _als_score_correct(als_model, u_index: int, m_index: int) -> float:
    return float(np.dot(als_model.item_factors[u_index], als_model.user_factors[m_index]))


def verify_als_score_convention(
    als_model, processed_dir: Path = PROCESSED_DIR, sample_size: int = 500, tol: float = 1e-6
) -> None:
    """Confirm the frozen hybrid model's als_score feature was built with the
    training (buggy/inverted) convention, not the corrected one, by
    recomputing both against a sample of real training rows."""
    sample = pd.read_parquet(
        processed_dir / "hybrid_train_pairs.parquet",
        columns=["u_index", "m_index", "als_score"],
    ).sample(n=sample_size, random_state=0)

    training_errs = []
    corrected_errs = []
    for row in sample.itertuples(index=False):
        stored = row.als_score
        training_errs.append(abs(_als_score_training_convention(als_model, row.u_index, row.m_index) - stored))
        corrected_errs.append(abs(_als_score_correct(als_model, row.u_index, row.m_index) - stored))

    mae_training = float(np.mean(training_errs))
    mae_corrected = float(np.mean(corrected_errs))
    assert mae_training < tol, (
        f"als_score training-convention mismatch, MAE={mae_training} — the frozen "
        f"hybrid model's als_score feature no longer matches the expected convention."
    )
    assert mae_corrected > tol, (
        "training and corrected als_score conventions were indistinguishable on this "
        "sample — verification is not meaningful, investigate before trusting it."
    )


def recall_at_k(ranked_ids: Sequence[int], positives: Set[int], k: int = K) -> float:
    if not positives:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(topk & positives) / len(positives)


def _dcg(ranked_ids: Sequence[int], positives: Set[int]) -> float:
    return sum(1.0 / np.log2(rank + 2) for rank, mid in enumerate(ranked_ids) if mid in positives)


def ndcg_at_k(ranked_ids: Sequence[int], positives: Set[int], k: int = K) -> float:
    if not positives:
        return 0.0
    topk = ranked_ids[:k]
    dcg = _dcg(topk, positives)
    ideal_hits = min(k, len(positives))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: Sequence[int], positives: Set[int], k: int = K) -> float:
    topk = set(ranked_ids[:k])
    return len(topk & positives) / k


def build_interaction_sets(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    train_items_by_user = train.groupby("u_index")["m_index"].apply(set).to_dict()
    test_items_by_user = test.groupby("u_index")["m_index"].apply(set).to_dict()
    return train_items_by_user, test_items_by_user


def build_candidate_groups(
    train_items_by_user: Dict[int, Set[int]],
    test_items_by_user: Dict[int, Set[int]],
    all_movie_ids: np.ndarray,
    n_negatives: int = N_NEGATIVES,
    seed: int = 42,
) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    groups: Dict[int, List[int]] = {}
    for u_index in sorted(test_items_by_user.keys()):
        positives = test_items_by_user[u_index]
        excluded = train_items_by_user.get(u_index, set()) | positives
        pool = np.setdiff1d(all_movie_ids, np.array(list(excluded), dtype=np.int64))
        if len(pool) == 0:
            warnings.warn(f"user {u_index} has no available negatives, skipping")
            continue
        n_neg = min(n_negatives, len(pool))
        if n_neg < n_negatives:
            warnings.warn(f"user {u_index} only has {len(pool)} available negatives (< {n_negatives})")
        sampled = rng.choice(pool, size=n_neg, replace=False)
        groups[u_index] = list(positives) + [int(m) for m in sampled]
    return groups


def rank_als_group(als_model, u_index: int, candidate_m_indices: Sequence[int]) -> List[int]:
    user_embedding = als_model.item_factors[u_index]
    movie_embeddings = als_model.user_factors[list(candidate_m_indices)]
    scores = movie_embeddings @ user_embedding
    order = sorted(range(len(candidate_m_indices)), key=lambda i: (-scores[i], candidate_m_indices[i]))
    return [candidate_m_indices[i] for i in order]


def build_movie_genre_matrix(movie_genres: pd.DataFrame, movie_map: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(movie_genres, columns=["genre"], prefix="", prefix_sep="")
    matrix = dummies.groupby("m_index").sum()
    matrix = matrix.reindex(columns=GENRE_ORDER, fill_value=0)
    matrix = matrix.reindex(movie_map["m_index"], fill_value=0)
    return matrix.astype(float)


def build_user_genre_profile(train: pd.DataFrame, movie_genre_matrix: pd.DataFrame) -> pd.DataFrame:
    merged = train[["u_index", "m_index"]].merge(
        movie_genre_matrix, left_on="m_index", right_index=True, how="left"
    )
    return merged.groupby("u_index")[GENRE_ORDER].mean()


def genre_similarity(user_vec: np.ndarray, movie_vec: np.ndarray) -> float:
    user_norm = np.linalg.norm(user_vec)
    movie_norm = np.linalg.norm(movie_vec)
    if user_norm == 0 or movie_norm == 0:
        return 0.0
    return float(np.dot(user_vec, movie_vec) / (user_norm * movie_norm))


def _als_score_training_convention_batch(als_model, u_index: int, m_indices: Sequence[int]) -> np.ndarray:
    return np.array([_als_score_training_convention(als_model, u_index, m) for m in m_indices])


def build_feature_matrix(
    u_index: int,
    candidate_m_indices: Sequence[int],
    user_genre_profile: pd.DataFrame,
    movie_genre_matrix: pd.DataFrame,
    als_model,
) -> np.ndarray:
    n = len(candidate_m_indices)
    if u_index in user_genre_profile.index:
        user_vec = user_genre_profile.loc[u_index].to_numpy()
    else:
        user_vec = np.zeros(len(GENRE_ORDER))

    movie_mat = movie_genre_matrix.loc[list(candidate_m_indices)].to_numpy()
    user_cols = np.tile(user_vec, (n, 1))
    genre_sim = np.array([genre_similarity(user_vec, movie_mat[i]) for i in range(n)])
    als_scores = _als_score_training_convention_batch(als_model, u_index, candidate_m_indices)

    return np.hstack([user_cols, movie_mat, genre_sim.reshape(-1, 1), als_scores.reshape(-1, 1)])


def rank_hybrid_group(
    hybrid_model,
    u_index: int,
    candidate_m_indices: Sequence[int],
    user_genre_profile: pd.DataFrame,
    movie_genre_matrix: pd.DataFrame,
    als_model,
) -> List[int]:
    X = build_feature_matrix(u_index, candidate_m_indices, user_genre_profile, movie_genre_matrix, als_model)
    scores = hybrid_model.predict(X)
    order = sorted(range(len(candidate_m_indices)), key=lambda i: (-scores[i], candidate_m_indices[i]))
    return [candidate_m_indices[i] for i in order]


def evaluate_model_for_seed(
    rank_fn: Callable[[int, List[int]], List[int]],
    candidate_groups: Dict[int, List[int]],
    test_items_by_user: Dict[int, Set[int]],
    k: int = K,
) -> Dict[str, float]:
    recalls, ndcgs, precisions = [], [], []
    for u_index, group in candidate_groups.items():
        ranked = rank_fn(u_index, group)
        positives = test_items_by_user[u_index]
        recalls.append(recall_at_k(ranked, positives, k))
        ndcgs.append(ndcg_at_k(ranked, positives, k))
        precisions.append(precision_at_k(ranked, positives, k))
    return {
        "recall_at_10": float(np.mean(recalls)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "precision_at_10": float(np.mean(precisions)),
    }


class SeedResult(NamedTuple):
    model: str
    seed: int
    recall_at_10: float
    ndcg_at_10: float
    precision_at_10: float


def run_unified_evaluation(
    processed_dir: Path = PROCESSED_DIR,
    k: int = K,
    n_negatives: int = N_NEGATIVES,
    seeds: List[int] = EVAL_SEEDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = load_artifacts(processed_dir)
    verify_factor_shapes(artifacts)
    verify_feature_column_order(processed_dir)
    verify_als_score_convention(artifacts.als_model, processed_dir)

    train_items_by_user, test_items_by_user = build_interaction_sets(artifacts.train, artifacts.test)
    movie_genre_matrix = build_movie_genre_matrix(artifacts.movie_genres, artifacts.movie_map)
    user_genre_profile = build_user_genre_profile(artifacts.train, movie_genre_matrix)
    all_movie_ids = artifacts.movie_map["m_index"].to_numpy()

    results: List[SeedResult] = []
    for seed in seeds:
        groups = build_candidate_groups(train_items_by_user, test_items_by_user, all_movie_ids, n_negatives, seed)

        als_metrics = evaluate_model_for_seed(
            lambda u, g: rank_als_group(artifacts.als_model, u, g), groups, test_items_by_user, k
        )
        results.append(SeedResult("ALS", seed, **als_metrics))

        hybrid_metrics = evaluate_model_for_seed(
            lambda u, g: rank_hybrid_group(
                artifacts.hybrid_model, u, g, user_genre_profile, movie_genre_matrix, artifacts.als_model
            ),
            groups,
            test_items_by_user,
            k,
        )
        results.append(SeedResult("Hybrid LightGBM", seed, **hybrid_metrics))

    raw_df = pd.DataFrame(results)
    summary_df = raw_df.groupby("model")[["recall_at_10", "ndcg_at_10", "precision_at_10"]].agg(["mean", "std"])
    return raw_df, summary_df


MODEL_LABELS = {
    "ALS": "ALS (factors=64, reg=0.1, iters=20)",
    "Hybrid LightGBM": "Hybrid LightGBM Ranker",
}


def format_markdown_table(summary_df: pd.DataFrame, k: int = K) -> str:
    lines = [
        f"| Model | Recall@{k} (mean±std) | NDCG@{k} (mean±std) | Precision@{k} (mean±std) |",
        "|---|---|---|---|",
    ]
    for model in ["ALS", "Hybrid LightGBM"]:
        row = summary_df.loc[model]
        label = MODEL_LABELS[model]
        recall = f"{row[('recall_at_10', 'mean')]:.4f} ± {row[('recall_at_10', 'std')]:.4f}"
        ndcg = f"{row[('ndcg_at_10', 'mean')]:.4f} ± {row[('ndcg_at_10', 'std')]:.4f}"
        precision = f"{row[('precision_at_10', 'mean')]:.4f} ± {row[('precision_at_10', 'std')]:.4f}"
        lines.append(f"| {label} | {recall} | {ndcg} | {precision} |")
    return "\n".join(lines)
