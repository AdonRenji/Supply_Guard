
# ============================================================
# SupplyGuard - Feature Redundancy Experiments
# ============================================================

import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

from pathlib import Path

from pathlib import Path

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "DataCoSupplyChainDataset.csv"
)
THRESHOLD = 0.41
RANDOM_STATE = 42

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("=" * 75)
print("SUPPLYGUARD - FEATURE REDUNDANCY EXPERIMENTS")
print("=" * 75)

df = pd.read_csv(DATASET_PATH, encoding="latin1")

print("Dataset loaded:", len(df), "rows")
print("Columns:", len(df.columns))

# ------------------------------------------------------------
# 3. CREATE DATE FEATURES
# Match the original training script
# ------------------------------------------------------------

df["DateOrders"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

df["Order Month"] = df["DateOrders"].dt.month
df["Order Day"] = df["DateOrders"].dt.day
df["Order Day of Week"] = df["DateOrders"].dt.dayofweek
df["Order Hour"] = df["DateOrders"].dt.hour

df["Is Weekend"] = (
    df["Order Day of Week"] >= 5
).astype(int)

df["Order Week of Year"] = (
    df["DateOrders"].dt.isocalendar().week.astype(int)
)

df["Order Day of Month"] = df["DateOrders"].dt.day

df["Is Month Start"] = (
    df["DateOrders"].dt.is_month_start
).astype(int)

df["Is Month End"] = (
    df["DateOrders"].dt.is_month_end
).astype(int)

df["Is Quarter Start"] = (
    df["DateOrders"].dt.is_quarter_start
).astype(int)

df["Is Quarter End"] = (
    df["DateOrders"].dt.is_quarter_end
).astype(int)

df["Is Business Hour"] = (
    (df["Order Day of Week"] < 5)
    & (df["Order Hour"] >= 9)
    & (df["Order Hour"] <= 17)
).astype(int)

df["Is Night"] = (
    (df["Order Hour"] < 6)
    | (df["Order Hour"] >= 22)
).astype(int)

df["Interaction_ShippingMode_OrderHour"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Order Hour"].astype(str)
)

# ------------------------------------------------------------
# 4. ORIGINAL 31 FEATURES
# ------------------------------------------------------------

features = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Customer Country",
    "Category Name",
    "Order Item Quantity",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Customer Segment",
    "Department Name",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "Sales",
    "Order Month",
    "Order Day",
    "Order Day of Week",
    "Order Hour",
    "Is Weekend",
    "Order Week of Year",
    "Order Day of Month",
    "Is Month Start",
    "Is Month End",
    "Is Quarter Start",
    "Is Quarter End",
    "Is Business Hour",
    "Is Night",
    "Interaction_ShippingMode_OrderHour",
    "Customer City",
    "Type",
    "Customer State"
]

target = "Late_delivery_risk"

categorical_features = [
    "Shipping Mode",
    "Customer Country",
    "Category Name",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Customer Segment",
    "Department Name",
    "Interaction_ShippingMode_OrderHour",
    "Customer City",
    "Type",
    "Customer State"
]

numerical_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "Sales",
    "Order Month",
    "Order Day",
    "Order Day of Week",
    "Order Hour",
    "Is Weekend",
    "Order Week of Year",
    "Order Day of Month",
    "Is Month Start",
    "Is Month End",
    "Is Quarter Start",
    "Is Quarter End",
    "Is Business Hour",
    "Is Night"
]

# ------------------------------------------------------------
# 5. CANDIDATE EXPERIMENTS
# Each experiment removes a specified feature or group.
# ------------------------------------------------------------

experiments = {
    "Baseline - All 31": [],

    "Remove duplicate Order Day": [
        "Order Day"
    ],

    "Remove Is Weekend": [
        "Is Weekend"
    ],

    "Remove Is Night": [
        "Is Night"
    ],

    "Remove Interaction Feature": [
        "Interaction_ShippingMode_OrderHour"
    ],

    "Remove Order Month": [
        "Order Month"
    ],

    "Remove Order Week of Year": [
        "Order Week of Year"
    ],

    "Remove three derived features": [
        "Order Day",
        "Is Weekend",
        "Is Night"
    ]
}

# ------------------------------------------------------------
# 6. SPLIT DATA ONCE
# All experiments use identical test records.
# ------------------------------------------------------------

X = df[features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Classification threshold:", THRESHOLD)

# ------------------------------------------------------------
# 7. TRAIN AND EVALUATE AN EXPERIMENT
# ------------------------------------------------------------

def run_experiment(name, removed_features):

    selected_features = [
        f for f in features
        if f not in removed_features
    ]

    categorical = [
        f for f in categorical_features
        if f in selected_features
    ]

    numerical = [
        f for f in numerical_features
        if f in selected_features
    ]

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numerical
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        )
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    print(f"\nTraining: {name}")
    print("Features:", len(selected_features))
    print("Removed:", removed_features or "None")

    start_time = time.time()

    model.fit(
        X_train[selected_features],
        y_train
    )

    probabilities = model.predict_proba(
        X_test[selected_features]
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    elapsed = time.time() - start_time

    result = {
        "Experiment": name,
        "Feature Count": len(selected_features),
        "Removed Features": ", ".join(removed_features) or "None",
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "Precision (%)": round(
            precision_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "Recall (%)": round(
            recall_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "F1 Score (%)": round(
            f1_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "ROC-AUC (%)": round(
            roc_auc_score(y_test, probabilities) * 100, 2
        ),
        "Training Time (seconds)": round(elapsed, 2)
    }

    print(
        "Accuracy:", result["Accuracy (%)"],
        "| Recall:", result["Recall (%)"],
        "| F1:", result["F1 Score (%)"],
        "| ROC-AUC:", result["ROC-AUC (%)"]
    )

    return result

# ------------------------------------------------------------
# 8. RUN ALL EXPERIMENTS
# ------------------------------------------------------------

results = []

for name, removed_features in experiments.items():
    results.append(
        run_experiment(name, removed_features)
    )

# ------------------------------------------------------------
# 9. DISPLAY AND SAVE RESULTS
# ------------------------------------------------------------

comparison = pd.DataFrame(results)

baseline = comparison.iloc[0]

for metric in [
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1 Score (%)",
    "ROC-AUC (%)"
]:
    comparison[metric + " Change"] = (
        comparison[metric] - baseline[metric]
    ).round(2)

print("\n" + "=" * 100)
print("FINAL FEATURE COMPARISON")
print("=" * 100)

print(comparison.to_string(index=False))

output_path = "feature_redundancy_results.csv"

comparison.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
print("Existing saved model files were not modified.")
