import pandas as pd


def merge_and_clean(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    rating_min: float = 0.5,
    rating_max: float = 5.0,
) -> pd.DataFrame:
    # Merge
    df = ratings.merge(movies, on="movie_id", how="left")

    # Basic cleaning
    df = df.dropna(subset=["title", "genres"])
    df = df[(df["rating"] >= rating_min) & (df["rating"] <= rating_max)]

    # Types
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["genres"] = df["genres"].fillna("(no genres listed)")

    # Stable integer ids
    df["u_index"] = pd.factorize(df["user_id"])[0]
    df["m_index"] = pd.factorize(df["movie_id"])[0]

    cols = [
        "user_id",
        "movie_id",
        "u_index",
        "m_index",
        "rating",
        "timestamp",
        "title",
        "genres",
    ]
    return df[cols]


def build_genre_table(df: pd.DataFrame) -> pd.DataFrame:
    # One row per movie
    movies = df[["m_index", "genres"]].drop_duplicates().copy()
    movies["genres"] = movies["genres"].str.split("|")
    exploded = movies.explode("genres", ignore_index=True)
    exploded = exploded.rename(columns={"genres": "genre"})
    return exploded[["m_index", "genre"]]