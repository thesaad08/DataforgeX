import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer
)

def scale_data(
    df,
    method="standard",
    cols=None
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    df = df.copy()

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns

    if len(cols) == 0:
        raise ValueError("No numeric columns to scale")

    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    elif method == "robust":
        scaler = RobustScaler()

    elif method == "normalize":
        scaler = Normalizer()

    else:
        raise ValueError("method must be standard, minmax, robust, or normalize")

    df[cols] = scaler.fit_transform(df[cols])

    return df
