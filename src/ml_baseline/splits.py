from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def random_split(
    df: pd.DataFrame,
    *,
    target: str,
    test_size: float,
    seed: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target]
    strat = y if stratify else None
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def time_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if time_col not in df.columns:
        raise ValueError(f"time_col not found: {time_col}")
    out = df.sort_values(time_col).reset_index(drop=True)
    n_test = max(1, int(len(out) * test_size))
    test = out.iloc[-n_test:]
    train = out.iloc[:-n_test]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def group_split(
    df: pd.DataFrame,
    *,
    group_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"group_col not found: {group_col}")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[group_col]
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    return train.reset_index(drop=True), test.reset_index(drop=True)
