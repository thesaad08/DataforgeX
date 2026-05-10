import pandas as pd
from sklearn.preprocessing import LabelEncoder

def auto_encode(
    df,
    method="auto",
    onehot_threshold=10,
    drop_first=True
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    df = df.copy()
    encoded_df = df.copy()

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    for col in cat_cols:
        unique_vals = df[col].nunique(dropna=True)

        # ---- Auto mode ----
        if method == "auto":
            if unique_vals <= onehot_threshold:
                dummies = pd.get_dummies(
                    df[col],
                    prefix=col,
                    drop_first=drop_first
                )
                encoded_df.drop(columns=[col], inplace=True)
                encoded_df = pd.concat([encoded_df, dummies], axis=1)
            else:
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(df[col].astype(str))

        # ---- Force OneHot ----
        elif method == "onehot":
            dummies = pd.get_dummies(
                df[col],
                prefix=col,
                drop_first=drop_first
            )
            encoded_df.drop(columns=[col], inplace=True)
            encoded_df = pd.concat([encoded_df, dummies], axis=1)

        # ---- Force Label ----
        elif method == "label":
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col].astype(str))

        else:
            raise ValueError("method must be auto, onehot, or label")

    return encoded_df
