import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_outliers(
    df,
    method="iqr",
    z_thresh=3.0,
    contamination=0.05
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    num_df = df.select_dtypes(include=[np.number])

    if num_df.empty:
        raise ValueError("No numeric columns found")

    mask = pd.DataFrame(False, index=df.index, columns=num_df.columns)

    if method == "iqr":
        Q1 = num_df.quantile(0.25)
        Q3 = num_df.quantile(0.75)
        IQR = Q3 - Q1
        mask = (num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))

    elif method == "zscore":
        z = (num_df - num_df.mean()) / num_df.std(ddof=0)
        mask = z.abs() > z_thresh

    elif method == "iforest":
        iso = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        preds = iso.fit_predict(num_df)
        mask.loc[:, :] = (preds == -1)[:, None]

    else:
        raise ValueError("method must be iqr, zscore, or iforest")

    return mask


def remove_outliers(
    df,
    method="iqr",
    z_thresh=3.0,
    contamination=0.05
):
    mask = detect_outliers(
        df,
        method=method,
        z_thresh=z_thresh,
        contamination=contamination
    )

    return df.loc[~mask.any(axis=1)].reset_index(drop=True)


def cap_outliers(
    df,
    method="iqr",
    z_thresh=3.0
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    if method == "iqr":
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[num_cols] = df[num_cols].clip(lower, upper, axis=1)

    elif method == "zscore":
        mean = df[num_cols].mean()
        std = df[num_cols].std(ddof=0)

        lower = mean - z_thresh * std
        upper = mean + z_thresh * std

        df[num_cols] = df[num_cols].clip(lower, upper, axis=1)

    else:
        raise ValueError("method must be iqr or zscore")

    return df
