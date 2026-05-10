# 🚀 DataForgeX – Code Reduction Examples

This document shows how **DataForgeX (DFX)** reduces long and repetitive
data science code into **simple and readable one-liners**.

---

## 1. Handling Missing Values

### Without DataForgeX

    import pandas as pd

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

### With DataForgeX

    from dfx import handle_missing_values
    df = handle_missing_values(df)

Code reduced: ~10 lines → 1 line

---

## 2. Fixing Data Types Automatically

### Without DataForgeX

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["flag"] = df["flag"].map({"yes": 1, "no": 0})

### With DataForgeX

    from dfx import auto_fix_dtypes
    df = auto_fix_dtypes(df)

---

## 4. Outlier Handling

### Without DataForgeX

    Q1 = df["chol"].quantile(0.25)
    Q3 = df["chol"].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df["chol"] >= Q1 - 1.5 * IQR) &
        (df["chol"] <= Q3 + 1.5 * IQR)
    ]

### With DataForgeX

    from dfx import remove_outliers
    df = remove_outliers(df)

---

## 5. Encoding Categorical Features

### Without DataForgeX

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df["sex"] = le.fit_transform(df["sex"])
    df = pd.get_dummies(df, columns=["cp"], drop_first=True)

### With DataForgeX

    from dfx import auto_encode
    df = auto_encode(df)

---

## 6. Feature Scaling

### Without DataForgeX

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

### With DataForgeX

    from dfx import scale_data
    df = scale_data(df)

---

## 7. Model Evaluation

### Without DataForgeX

    from sklearn.metrics import accuracy_score, classification_report

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

### With DataForgeX

    from dfx import evaluate_classification
    evaluate_classification(model, X_test, y_test)

---

## 8. Full Workflow Comparison

### Without DataForgeX

    df = fix_types(df)
    df = fill_missing(df)
    plot_all(df)
    df = encode(df)
    df = scale(df)
    evaluate(model)

### With DataForgeX

    from dfx import auto_fix_dtypes, handle_missing_values, quick_eda
    from dfx import auto_encode, scale_data, evaluate_model

    df = auto_fix_dtypes(df)
    df = handle_missing_values(df)
    quick_eda(df)
    df = auto_encode(df)
    df = scale_data(df)
    evaluate_model(model, X_test, y_test)

---


# 📦 Installation

Install the latest stable version from PyPI:

```bash
pip install dfx
```

Install with optional ML features:

```bash
pip install dfx[ml]
```

Install development tools:

```bash
pip install dfx[dev]
```

---

# 🚀 Quick Start

```python
import pandas as pd
from dfx import (
    auto_fix_dtypes,
    handle_missing_values,
    auto_encode,
    scale_data,
    evaluate_classification
)

# Load data
df = pd.read_csv("data.csv")

# Preprocess
df = auto_fix_dtypes(df)
df = handle_missing_values(df)
df = auto_encode(df)
df = scale_data(df)

# Train model
from sklearn.ensemble import RandomForestClassifier

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier().fit(X, y)

# Evaluate
evaluate_classification(model, X, y)
```

---

# 🧠 Core Features

| Feature | Description |
|--------|------------|
| Automatic dtype fixing | Detects numeric, date, and boolean columns |
| Missing value handling | Smart median/mode imputation |
| Auto encoding | Label + one-hot encoding |
| Scaling | Standard/robust scaling |
| Outlier removal | IQR-based filtering |
| Quick EDA | Summary + plots |
| Model evaluation | Metrics + reports |

---

# 🏗️ Library Design

DataForgeX follows three design principles:

- **Automation** — minimal user code  
- **Consistency** — standardized preprocessing  
- **Compatibility** — works with pandas & sklearn  

It integrates directly into existing ML pipelines without replacing them.

---

# 📊 Typical Workflow

```
Raw Data
   ↓
DataForgeX Preprocessing
   ↓
Clean Dataset
   ↓
Model Training
   ↓
Evaluation
```

---

# 📁 Project Structure

```
DataForgeX/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
└── src/
    └── dfx/
        ├── preprocessing.py
        ├── encoding.py
        ├── scaling.py
        ├── outliers.py
        ├── eda.py
        └── evaluation.py
```

---

# 🎯 Use Cases

DataForgeX is useful for:

- Machine learning preprocessing  
- Kaggle competitions  
- Research experiments  
- Student projects  
- Rapid prototyping  
- Data cleaning automation  

---

# 🤝 Contributing

Contributions are welcome.

1. Fork repository  
2. Create feature branch  
3. Commit changes  
4. Submit pull request  

---


## Final Summary

- Less code
- Better readability
- Faster development
- Reusable functions

DataForgeX lets you focus on logic, not boilerplate.

---
## Authors and Contributors

### Author and Lead Developer
**Prince Saxena**  
Creator and primary developer of DataForgeX.  
Responsible for the design, architecture, and implementation of the library.

### Contributors
The development of DataForgeX benefited from valuable collaboration and input from:

- **Mohd Saad Sherwani** — Conceptual discussions and feature ideation  
- **Hammad Rafeeque** — Testing, feedback, and usability evaluation  

Their support and insights helped refine the direction and usability of the project.

---

# 📜 License

MIT License © 2026 Prince Saxena