
# ============================================================
# SupplyGuard - Customer Location Feature Comparison
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
    roc_auc_score,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

DATASET_PATH = "DataCoSupplyChainDataset.csv"
THRESHOLD = 0.41
RANDOM_STATE = 42

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("=" * 65)
print("SUPPLYGUARD - LOCATION FEATURE COMPARISON")
print("=" * 65)

df = pd.read_csv(DATASET_PATH, encoding="latin1")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ------------------------------------------------------------
# 3. CREATE DATE FEATURES
# Match train_final_model.py
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
# 4. DEFINE THE ORIGINAL 31 FEATURES
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

# Model B removes only the three customer-location features.
location_features = [
    "Customer Country",
    "Customer City",
    "Customer State"
]

reduced_features = [
    feature for feature in features
    if feature not in location_features
]

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
# 5. PREPARE DATA AND SPLIT ONCE
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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model A features:", len(features))
print("Model B features:", len(reduced_features))

# ------------------------------------------------------------
# 6. CREATE AND EVALUATE EACH MODEL
# ------------------------------------------------------------

def evaluate_model(model_name, selected_features):
    print(f"\nTraining {model_name}...")
    start_time = time.time()

    categorical = [
        col for col in categorical_features
        if col in selected_features
    ]

    numerical = [
        col for col in numerical_features
        if col in selected_features
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical),
        ("cat", categorical_transformer, categorical)
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

    results = {
        "Model": model_name,
        "Feature Count": len(selected_features),
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

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return results


# ------------------------------------------------------------
# 7. COMPARE BOTH MODELS
# ------------------------------------------------------------

results_a = evaluate_model(
    "Model A - All 31 Features",
    features
)

results_b = evaluate_model(
    "Model B - Without Customer Location",
    reduced_features
)

comparison = pd.DataFrame([results_a, results_b])

print("\n" + "=" * 90)
print("FINAL COMPARISON")
print("=" * 90)
print(comparison.to_string(index=False))

comparison.to_csv(
    "location_feature_comparison.csv",
    index=False
)

print("\nResults saved to location_feature_comparison.csv")
print("Existing saved model files were not modified.")
