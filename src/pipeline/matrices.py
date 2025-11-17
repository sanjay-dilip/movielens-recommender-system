from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_item_user_matrix(df: pd.DataFrame) -> sp.csr_matrix:
    n_items = df["m_index"].max() + 1
    n_users = df["u_index"].max() + 1

    coo = sp.coo_matrix(
        (
            df["rating"].astype(np.float32),
            (df["m_index"], df["u_index"]),
        ),
        shape=(n_items, n_users),
    )

    return coo.tocsr()


def save_sparse(matrix: sp.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix)


def build_user_map(df: pd.DataFrame) -> pd.DataFrame:
    return df[["user_id", "u_index"]].drop_duplicates().reset_index(drop=True)


def build_movie_map(df: pd.DataFrame) -> pd.DataFrame:
    return df[["movie_id", "m_index", "title", "genres"]].drop_duplicates().reset_index(drop=True)