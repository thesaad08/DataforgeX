import pandas as pd
import numpy as np
import re

def auto_fix_dtypes(df, cat_threshold=0.05, date_threshold=0.6):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    df = df.copy()

    for col in df.columns:
        series = df[col]

        if series.dtype != "object":
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        # ---- Boolean check ----
        bool_map = {
            "true": True, "false": False,
            "yes": True, "no": False,
            "1": True, "0": False
        }

        lowered = non_null.astype(str).str.lower()
        if lowered.isin(bool_map.keys()).mean() > 0.9:
            df[col] = series.astype(str).str.lower().map(bool_map)
            continue

        # ---- Numeric check ----
        num_conv = pd.to_numeric(series, errors="coerce")
        if num_conv.notna().mean() > 0.8:
            df[col] = num_conv
            continue

        # ---- Date check ----
        date_conv = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        if date_conv.notna().mean() > date_threshold:
            df[col] = date_conv
            continue

        # ---- Category check ----
        unique_ratio = non_null.nunique() / len(non_null)
        if unique_ratio <= cat_threshold:
            df[col] = series.astype("category")
            continue

        # ---- Clean string ----
        df[col] = series.astype(str).str.strip()

    return df
