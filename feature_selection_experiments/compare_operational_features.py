
# ============================================================
# SupplyGuard - Operational Feature Selection Experiment
# ============================================================

import time
from pathlib import Path

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

# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "DataCoSupplyChainDataset.csv"

OUTPUT_DIR = BASE_DIR / "results" / "feature_group_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "operational_feature_comparison.csv"

THRESHOLD = 0.41
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("SUPPLYGUARD - OPERATIONAL FEATURE EXPERIMENT")
print("=" * 70)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH, encoding="latin1")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ============================================================
# 3. CREATE REQUIRED DATE FEATURES
# ============================================================

df["DateOrders"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

df["Order Month"] = df["DateOrders"].dt.month
df["Order Day of Week"] = df["DateOrders"].dt.dayofweek
df["Order Hour"] = df["DateOrders"].dt.hour

# ============================================================
# 4. SELECT EIGHT OPERATIONAL FEATURES
# ============================================================

features = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Order Country",
    "Order State",
    "Order Region",
    "Order Month",
    "Order Day of Week",
    "Order Hour"
]

target = "Late_delivery_risk"

required_columns = features + [target]
missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Exclude rows with an invalid target.
df = df.dropna(subset=[target]).copy()

if not set(df[target].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

X = df[features].copy()
y = df[target].astype(int).copy()

categorical_features = [
    "Shipping Mode",
    "Order Country",
    "Order State",
    "Order Region"
]

numerical_features = [
    "Days for shipment (scheduled)",
    "Order Month",
    "Order Day of Week",
    "Order Hour"
]

print("\nSelected features:")
for feature in features:
    print("-", feature)

print("\nTotal selected features:", len(features))

# ============================================================
# 5. SPLIT DATA ONCE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ============================================================
# 6. PREPROCESSING AND MODEL
# ============================================================

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numerical_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
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

# ============================================================
# 7. TRAIN AND EVALUATE
# ============================================================

print("\nTraining operational model...")
start_time = time.time()

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

elapsed = time.time() - start_time

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

results = {
    "Model": "Eight Operational Features",
    "Feature Count": len(features),
    "Accuracy (%)": round(accuracy_score(y_test, predictions) * 100, 2),
    "Precision (%)": round(
        precision_score(y_test, predictions, zero_division=0) * 100, 2
    ),
    "Recall (%)": round(
        recall_score(y_test, predictions, zero_division=0) * 100, 2
    ),
    "F1 Score (%)": round(
        f1_score(y_test, predictions, zero_division=0) * 100, 2
    ),
    "ROC-AUC (%)": round(roc_auc_score(y_test, probabilities) * 100, 2),
    "Training Time (seconds)": round(elapsed, 2),
    "True Negatives": int(cm[0, 0]),
    "False Positives": int(cm[0, 1]),
    "False Negatives": int(cm[1, 0]),
    "True Positives": int(cm[1, 1])
}

print("\nConfusion matrix:")
print(cm)

print("\nResults:")
for key, value in results.items():
    print(f"{key}: {value}")

pd.DataFrame([results]).to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nExisting saved model and Flask app were not modified.")
print("=" * 70)
