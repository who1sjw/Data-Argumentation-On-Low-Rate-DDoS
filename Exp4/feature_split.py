import pandas as pd
import numpy as np


def split_features(df: pd.DataFrame, discrete_unique_threshold: int = 10):
    """
    Split DataFrame columns into continuous and discrete based on dtype and unique value counts.

    Parameters:
    - df: pd.DataFrame containing feature columns (excluding target).
    - discrete_unique_threshold: if an integer column has <= this many unique values, treat as discrete.

    Returns:
    - continuous_cols: List[str] of continuous feature names.
    - discrete_cols:  List[str] of discrete feature names.
    """
    # Normalize column names
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    continuous_cols = []
    discrete_cols = []
    for col in df_clean.columns:
        # Skip target column if present
        if col.lower() == 'label':
            continue
        series = df_clean[col]
        # Float dtype => continuous
        if pd.api.types.is_float_dtype(series):
            continuous_cols.append(col)
        # Integer dtype: decide by unique count
        elif pd.api.types.is_integer_dtype(series):
            if series.nunique() <= discrete_unique_threshold:
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
        # Other types (object, bool): treat as discrete
        else:
            discrete_cols.append(col)
    return continuous_cols, discrete_cols


def get_feature_indices(df: pd.DataFrame, continuous_cols: list, discrete_cols: list):
    """
    Convert column names to integer indices for numpy slicing.

    Returns:
    - idx_cont: List[int] indices for continuous columns
    - idx_disc: List[int] indices for discrete columns
    """
    # Normalize column names
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()
    cols = list(df_clean.columns)
    idx_cont = [cols.index(c) for c in continuous_cols]
    idx_disc = [cols.index(c) for c in discrete_cols]
    return idx_cont, idx_disc
