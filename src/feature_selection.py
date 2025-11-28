import numpy as np
import pandas as pd


def auto_select_features(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    max_numeric: int = 20,
    max_categorical: int = 20,
):
    """
    I automatically pick a subset of numeric and categorical columns.

    - I ignore columns that are >95% missing.
    - I treat floats as numeric.
    - I treat objects/categories as categorical.
    - I treat low-cardinality integer columns as categorical (<=20 unique).
    - I return up to max_numeric + max_categorical columns to keep things manageable.
    """
    ignore_cols = {target_col, "TransactionID"}

    # drop mostly-missing columns
    na_frac = df.isna().mean()
    kept_cols = na_frac[na_frac < 0.95].index.tolist()
    kept_cols = [c for c in kept_cols if c not in ignore_cols]

    numeric_cols = []
    categorical_cols = []

    dtypes = df[kept_cols].dtypes

    for col in kept_cols:
        dt = dtypes[col]

        if dt == "object" or str(dt) == "category":
            categorical_cols.append(col)

        elif np.issubdtype(dt, np.integer):
            # treat low-cardinality integer columns as categorical
            if df[col].nunique(dropna=True) <= 20:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

        elif np.issubdtype(dt, np.floating):
            numeric_cols.append(col)

    # Trim to desired max
    numeric_cols = numeric_cols[:max_numeric]
    categorical_cols = categorical_cols[:max_categorical]

    print(f"Selected {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features.")
    print("Numeric:", numeric_cols)
    print("Categorical:", categorical_cols)

    return numeric_cols, categorical_cols