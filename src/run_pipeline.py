import sys
from pathlib import Path

# 1. Point ROOT to the project folder: Movie Recommender/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline.io import load_movielens, save_csv
from src.pipeline.processing import merge_and_clean, build_genre_table
from src.pipeline.splits import time_based_split
from src.pipeline.matrices import (
    build_item_user_matrix,
    save_sparse,
    build_user_map,
    build_movie_map,
)


def main():
    # 2. Raw and processed paths under the project root
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", ROOT)
    print("Raw dir:", raw_dir)
    print("Processed dir:", processed_dir)

    print("Loading raw MovieLens data...")
    ratings, movies = load_movielens(raw_dir)

    print("Merging and cleaning...")
    merged = merge_and_clean(ratings, movies)

    print("Building genre table...")
    genre_table = build_genre_table(merged)

    print("Splitting train and test...")
    train, test = time_based_split(merged, test_ratio=0.2, min_events_per_user=5)

    print("Building item x user train matrix...")
    item_user_train = build_item_user_matrix(train)

    print("Building id maps...")
    user_map = build_user_map(merged)
    movie_map = build_movie_map(merged)

    print("Saving CSVs and matrix...")
    save_csv(merged, processed_dir / "merged.csv")
    save_csv(genre_table, processed_dir / "movie_genres.csv")
    save_csv(train, processed_dir / "train.csv")
    save_csv(test, processed_dir / "test.csv")
    save_csv(user_map, processed_dir / "user_map.csv")
    save_sparse(item_user_train, processed_dir / "item_user_train.npz")

    # movie_map is useful to look up titles
    save_csv(movie_map, processed_dir / "movie_map.csv")

    print("✅ Pipeline complete. Outputs in:", processed_dir)


if __name__ == "__main__":
    main()
