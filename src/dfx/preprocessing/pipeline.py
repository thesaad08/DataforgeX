import pandas as pd

from dfx.preprocessing.autotype import auto_fix_dtypes
from dfx.preprocessing.encoding import auto_encode
from dfx.preprocessing.outliers import remove_outliers, cap_outliers
from dfx.preprocessing.scaling import scale_data


def preprocess_pipeline(
    df,
    encode=True,
    encoding_method="auto",
    handle_outliers="cap",   # "cap", "remove", None
    outlier_method="iqr",
    scaling=True,
    scaling_method="standard"
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    df = df.copy()

    # ---- Step 1: Fix data types ----
    df = auto_fix_dtypes(df)

    # ---- Step 2: Outlier handling ----
    if handle_outliers == "remove":
        df = remove_outliers(df, method=outlier_method)

    elif handle_outliers == "cap":
        df = cap_outliers(df, method=outlier_method)

    # ---- Step 3: Encoding ----
    if encode:
        df = auto_encode(df, method=encoding_method)

    # ---- Step 4: Scaling ----
    if scaling:
        df = scale_data(df, method=scaling_method)

    return df
