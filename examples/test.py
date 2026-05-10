# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from dfx import (
#     evaluate_classification, 
#     plot_confusion_matrix, 
#     plot_residuals, 
#     plot_correlation, 
#     detect_outliers, 
#     cap_outliers, 
#     remove_outliers, 
#     suggest_fill_strategy,
#     handle_missing_values,
#     plot_boxplots,
#     plot_correlation_heatmap
# )
# df = pd.read_csv("sample_data/heart.csv") 

##################### Model Eval ########################

# X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df["target"], test_size=0.2, random_state=42)

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# res = evaluate_classification(model, X_test, y_test)
# # plot_confusion_matrix(model, X_test, y_test) 
# plot_correlation(df,method="spearman",show_values=True)   
# # print(res)


################### Preprocessing ######################

# print(df.info())
# res = detect_outliers(df, method="iforest", contamination=0.1)
# print(res.sum())
# print(df.describe())
# r = remove_outliers(df, method="iqr")
# print(r.describe())


#################### Cleaning #########################
# data = pd.read_csv("sample_data/test_data.csv")
# print(data)
# print()
# print(suggest_fill_strategy(data))
# print()
# print(handle_missing_values(data))

# Visualization

# print(plot_correlation_heatmap(df,theme='corporate'))
# print(plot_correlation_heatmap(df))




############## Handling Missing Values ################
# Without DFX

# import pandas as pd

# for col in df.columns:
#     if df[col].dtype == "object":
#         df[col] = df[col].fillna(df[col].mode()[0])
#     else:
#         df[col] = df[col].fillna(df[col].median())


# With DFX

# from dataforge import handle_missing_values

# df = handle_missing_values(df)

################# Fixing Data Types ####################
# Without DFX
# df["age"] = pd.to_numeric(df["age"], errors="coerce")
# df["date"] = pd.to_datetime(df["date"], errors="coerce")
# df["flag"] = df["flag"].map({"yes": 1, "no": 0})


# With DFX

# from dataforge import auto_fix_dtypes

# df = auto_fix_dtypes(df)

