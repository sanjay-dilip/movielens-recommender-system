import pandas as pd
from typing import Tuple


def time_based_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_events_per_user: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Filter out very sparse users
    counts = df.groupby("u_index").size()
    keep_users = counts[counts >= min_events_per_user].index
    df = df[df["u_index"].isin(keep_users)].copy()

    # Sort by time
    df = df.sort_values(["u_index", "timestamp"])

    def split_one_user(group: pd.DataFrame):
        n = len(group)
        k = max(1, int(n * test_ratio))
        return group.iloc[:-k], group.iloc[-k:]

    trains = []
    tests = []

    for _, g in df.groupby("u_index"):
        tr, te = split_one_user(g)
        trains.append(tr)
        tests.append(te)

    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)

    return train, test