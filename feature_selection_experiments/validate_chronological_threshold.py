
import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

DATA_PATH = "DataCoSupplyChainDataset.csv"

FEATURES = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Order Country",
    "Order State",
    "Order Region",
    "Order Month",
    "Order Day of Week",
    "Order Hour",
]

TARGET = "Late_delivery_risk"

# Load and prepare data
df = pd.read_csv(DATA_PATH, encoding="latin1")

df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"], errors="coerce"
)

df = df.dropna(
    subset=["order date (DateOrders)", TARGET]
).copy()

dates = df["order date (DateOrders)"]

df["Order Month"] = dates.dt.month
df["Order Day of Week"] = dates.dt.dayofweek
df["Order Hour"] = dates.dt.hour

df = df.sort_values("order date (DateOrders)").reset_index(drop=True)

# Chronological split: 60% training, 20% validation, 20% final test
n = len(df)
train_end = int(n * 0.60)
validation_end = int(n * 0.80)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:validation_end].copy()
test = df.iloc[validation_end:].copy()

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_val = validation[FEATURES]
y_val = validation[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)

print("=" * 70)
print("SUPPLYGUARD - CHRONOLOGICAL THRESHOLD VALIDATION")
print("=" * 70)
print(f"Training rows:   {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Final test rows: {len(test):,}")
print(
    "Validation period:",
    validation["order date (DateOrders)"].min(),
    "to",
    validation["order date (DateOrders)"].max(),
)
print(
    "Final test period:",
    test["order date (DateOrders)"].min(),
    "to",
    test["order date (DateOrders)"].max(),
)

# Preprocessing fitted only on training data
categorical = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical = [
    col for col in FEATURES if col not in categorical
]

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical,
    ),
    (
        "num",
        SimpleImputer(strategy="median"),
        numerical,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

print("\nTraining experimental model...")
start = time.time()
model.fit(X_train, y_train)
print(f"Training completed in {time.time() - start:.2f} seconds.")

# Choose threshold using validation data only
val_prob = model.predict_proba(X_val)[:, 1]

candidate_thresholds = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.41
]

MIN_RECALL = 0.90

threshold_results = []

for threshold in candidate_thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, val_pred, labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "Recall": recall_score(y_val, val_pred),
        "Precision": precision_score(
            y_val, val_pred, zero_division=0
        ),
        "Accuracy": accuracy_score(y_val, val_pred),
        "F1": f1_score(y_val, val_pred, zero_division=0),
        "False Alarms": fp,
        "Missed Delays": fn,
    })

results = pd.DataFrame(threshold_results)

print("\nVALIDATION RESULTS")
print(results.round(4).to_string(index=False))

eligible = results[results["Recall"] >= MIN_RECALL]

if eligible.empty:
    chosen_threshold = results.sort_values(
        "Recall", ascending=False
    ).iloc[0]["Threshold"]

    print(
        f"\nNo candidate reached {MIN_RECALL:.0%} recall "
        "on validation data."
    )
    print("Using the highest-recall candidate for evaluation only.")
else:
    # Among thresholds meeting the recall target, choose
    # the one with the fewest false alarms.
    chosen_threshold = eligible.sort_values(
        ["False Alarms", "Threshold"],
        ascending=[True, False],
    ).iloc[0]["Threshold"]

print(f"\nSelected validation threshold: {chosen_threshold:.2f}")

# Evaluate exactly once on the untouched later test period
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= chosen_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

print("\nUNTOUCHED FINAL TEST RESULTS")
print(f"Threshold: {chosen_threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, test_pred):.4f}")
print(f"F1:        {f1_score(y_test, test_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_prob):.4f}")
print(f"False alarms: {fp:,}")
print(f"Missed delays: {fn:,}")

print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print([[tn, fp], [fn, tp]])

print("\nExperiment complete. No saved final model or app was changed.")
