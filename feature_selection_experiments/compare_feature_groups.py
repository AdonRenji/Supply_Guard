
# ============================================================
# SupplyGuard - Feature Group Comparison Experiments
# ============================================================

import os
import time
import warnings
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
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

DATASET_PATH = "DataCoSupplyChainDataset.csv"
OUTPUT_FOLDER = os.path.join("results", "feature_group_experiments")

THRESHOLD = 0.41
RANDOM_STATE = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("=" * 65)
print("SUPPLYGUARD - FEATURE GROUP COMPARISON")
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
    df["order date (DateOrders)"], errors="coerce"
)

df["Order Month"] = df["DateOrders"].dt.month
df["Order Day"] = df["DateOrders"].dt.day
df["Order Day of Week"] = df["DateOrders"].dt.dayofweek
df["Order Hour"] = df["DateOrders"].dt.hour

df["Is Weekend"] = (
    df["Order Day of Week"] >= 5
).astype(int)

df["Order Week of Year"] = (
    df["DateOrders"].dt.isocalendar().week.astype(float)
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
    "Customer State",
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
    "Customer State",
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
    "Is Night",
]

# ------------------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required = features + [target, "order date (DateOrders)"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        "Missing dataset columns: " + ", ".join(missing)
    )

df = df.dropna(subset=[target]).copy()

X = df[features].copy()
y = df[target].astype(int).copy()

# ------------------------------------------------------------
# 6. ONE FIXED 80/20 STRATIFIED SPLIT
# ------------------------------------------------------------

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining rows:", len(X_train_all))
print("Testing rows:", len(X_test_all))
print("Classification threshold:", THRESHOLD)

# ------------------------------------------------------------
# 7. FEATURE GROUP EXPERIMENTS
# Each experiment removes selected features from the baseline.
# ------------------------------------------------------------

experiments = [
    ("Baseline - all 31 features", []),

    ("Remove duplicate Order Day",
     ["Order Day"]),

    ("Remove Is Weekend",
     ["Is Weekend"]),

    ("Remove Is Night",
     ["Is Night"]),

    ("Remove Is Business Hour",
     ["Is Business Hour"]),

    ("Remove Month and Week",
     ["Order Month", "Order Week of Year"]),

    ("Remove duplicate day and weekend",
     ["Order Day", "Is Weekend"]),

    ("Remove duplicate day, weekend and night",
     ["Order Day", "Is Weekend", "Is Night"]),

    ("Remove customer location group",
     ["Customer Country", "Customer City", "Customer State"]),

    ("Remove Market and Order Region",
     ["Market", "Order Region"]),

    ("Remove Market and Order Country",
     ["Market", "Order Country"]),

    ("Remove Category and Department",
     ["Category Name", "Department Name"]),

    ("Remove Order Country and Order State",
     ["Order Country", "Order State"]),

    ("Remove interaction feature",
     ["Interaction_ShippingMode_OrderHour"]),

    ("Remove month, week and duplicate day",
     ["Order Month", "Order Week of Year", "Order Day"]),

    ("Remove duplicate day and all three flags",
     [
         "Order Day",
         "Is Weekend",
         "Is Night",
         "Is Business Hour",
     ]),
]

# ------------------------------------------------------------
# 8. MODEL PIPELINE
# ------------------------------------------------------------

def create_model(active_features):

    active_numeric = [
        col for col in numerical_features
        if col in active_features
    ]

    active_categorical = [
        col for col in categorical_features
        if col in active_features
    ]

    transformers = []

    if active_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        transformers.append((
            "num",
            numeric_pipeline,
            active_numeric,
        ))

    if active_categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore"
            )),
        ])

        transformers.append((
            "cat",
            categorical_pipeline,
            active_categorical,
        ))

    preprocessor = ColumnTransformer(transformers=transformers)

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", random_forest),
    ])

    return model

# ------------------------------------------------------------
# 9. TRAIN AND EVALUATE EVERY EXPERIMENT
# ------------------------------------------------------------

results = []

for number, (experiment_name, removed_features) in enumerate(
    experiments, start=1
):

    print("\n" + "=" * 65)
    print(f"EXPERIMENT {number}/{len(experiments)}")
    print(experiment_name)

    active_features = [
        col for col in features
        if col not in removed_features
    ]

    print("Features retained:", len(active_features))
    print("Features removed:", removed_features or "None")

    model = create_model(active_features)

    X_train = X_train_all[active_features]
    X_test = X_test_all[active_features]

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    results.append({
        "Experiment": experiment_name,
        "Features Retained": len(active_features),
        "Features Removed": ", ".join(removed_features) or "None",
        "Threshold": THRESHOLD,
        "Accuracy (%)": accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1 Score (%)": f1 * 100,
        "ROC-AUC (%)": auc * 100,
        "True Negative": int(cm[0, 0]),
        "False Positive": int(cm[0, 1]),
        "False Negative": int(cm[1, 0]),
        "True Positive": int(cm[1, 1]),
        "Training Time (seconds)": training_time,
    })

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc * 100:.2f}%")
    print("Confusion matrix:")
    print(cm)

# ------------------------------------------------------------
# 10. SAVE RESULTS SEPARATELY
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

baseline = results_df[
    results_df["Experiment"] == "Baseline - all 31 features"
].iloc[0]

for metric in [
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1 Score (%)",
    "ROC-AUC (%)",
]:
    results_df["Change vs Baseline - " + metric] = (
        results_df[metric] - baseline[metric]
    )

output_csv = os.path.join(
    OUTPUT_FOLDER,
    "feature_group_comparison.csv",
)

ranked_csv = os.path.join(
    OUTPUT_FOLDER,
    "feature_group_comparison_ranked.csv",
)

results_df.to_csv(output_csv, index=False)

ranked_df = results_df.sort_values(
    by=["F1 Score (%)", "ROC-AUC (%)"],
    ascending=False,
)

ranked_df.to_csv(ranked_csv, index=False)

print("\n" + "=" * 100)
print("FINAL COMPARISON - RANKED BY F1 SCORE")
print("=" * 100)

print(ranked_df[[
    "Experiment",
    "Features Retained",
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1 Score (%)",
    "ROC-AUC (%)",
]].round(2).to_string(index=False))

print("\nResults saved to:")
print(output_csv)
print(ranked_csv)

print("\nExisting model and Flask app were not overwritten.")
print("Review the results before removing any features.")
print("=" * 65)
