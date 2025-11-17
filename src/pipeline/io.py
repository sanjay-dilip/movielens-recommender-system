from pathlib import Path
import pandas as pd


def load_movielens(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = raw_dir / "ratings.dat"
    movies_path = raw_dir / "movies.dat"

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
    )

    return ratings, movies


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
