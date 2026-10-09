"""Dataset split helpers."""

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """Split rows by time so validation/test observations occur after training."""
    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:val_end].copy(),
        ordered.iloc[val_end:].copy(),
    )


def user_disjoint_split(
    df: pd.DataFrame,
    user_col: str = "uid",
    train_frac: float = 0.80,
    seed: int = 42,
):
    """Hold out entire users for unseen-customer evaluation."""
    users = pd.Series(df[user_col].dropna().unique()).sample(frac=1, random_state=seed)
    cutoff = int(len(users) * train_frac)
    train_users = set(users.iloc[:cutoff])
    test_users = set(users.iloc[cutoff:])
    return (
        df[df[user_col].isin(train_users)].copy(),
        df[df[user_col].isin(test_users)].copy(),
    )
