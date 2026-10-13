
# ============================================================
# SupplyGuard - Chronological Destination Feature Comparison
# Model A: Eight operational features
# Model B: Six features, excluding Order Country and Order State
#
# This experiment does not modify saved_model or app.py.
# ============================================================

import time
from pathlib import Path

import pandas as pd

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

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "DataCoSupplyChainDataset.csv"

OUTPUT_DIR = BASE_DIR / "results" / "feature_group_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "destination_chronological_comparison.csv"

TARGET = "Late_delivery_risk"
DATE_COLUMN = "order date (DateOrders)"
THRESHOLD = 0.41
RANDOM_STATE = 42

FEATURE_SETS = {
    "Model A - Eight features": [
        "Shipping Mode",
        "Days for shipment (scheduled)",
        "Order Country",
        "Order State",
        "Order Region",
        "Order Month",
        "Order Day of Week",
        "Order Hour",
    ],
    "Model B - Six features": [
        "Shipping Mode",
        "Days for shipment (scheduled)",
        "Order Region",
        "Order Month",
        "Order Day of Week",
        "Order Hour",
    ],
}

CATEGORICAL = {
    "Shipping Mode",
    "Order Country",
    "Order State",
    "Order Region",
}


def evaluate_model(name, features, train_df, validation_df):
    categorical = [f for f in features if f in CATEGORICAL]
    numerical = [f for f in features if f not in CATEGORICAL]

    transformers = [
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numerical,
        )
    ]

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print(f"\nTraining {name}...")
    start = time.time()
    model.fit(train_df[features], train_df[TARGET].astype(int))
    probabilities = model.predict_proba(validation_df[features])[:, 1]
    elapsed = time.time() - start

    y_true = validation_df[TARGET].astype(int)
    predictions = (probabilities >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    return {
        "Model": name,
        "Feature Count": len(features),
        "Training Rows": len(train_df),
        "Validation Rows": len(validation_df),
        "Training Start": str(train_df["_date"].min()),
        "Training End": str(train_df["_date"].max()),
        "Validation Start": str(validation_df["_date"].min()),
        "Validation End": str(validation_df["_date"].max()),
        "Threshold": THRESHOLD,
        "Accuracy (%)": round(accuracy_score(y_true, predictions) * 100, 2),
        "Precision (%)": round(
            precision_score(y_true, predictions, zero_division=0) * 100, 2
        ),
        "Recall (%)": round(
            recall_score(y_true, predictions, zero_division=0) * 100, 2
        ),
        "F1 Score (%)": round(
            f1_score(y_true, predictions, zero_division=0) * 100, 2
        ),
        "ROC-AUC (%)": round(roc_auc_score(y_true, probabilities) * 100, 2),
        "Training Time (seconds)": round(elapsed, 2),
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "True Positives": int(cm[1, 1]),
    }


def main():
    print("=" * 72)
    print("SUPPLYGUARD - CHRONOLOGICAL FEATURE COMPARISON")
    print("=" * 72)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, encoding="latin1")

    df["_date"] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=["_date", TARGET]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()
    df = df.sort_values("_date").reset_index(drop=True)

    df["Order Month"] = df["_date"].dt.month
    df["Order Day of Week"] = df["_date"].dt.dayofweek
    df["Order Hour"] = df["_date"].dt.hour

    all_features = sorted({
        feature
        for feature_list in FEATURE_SETS.values()
        for feature in feature_list
    })
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Chronological 60/20/20 split.
    # This run uses only the first 60% for training and next 20% for
    # validation. The final 20% is not evaluated here.
    n = len(df)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()

    print(f"Usable rows: {n}")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Training period: {train_df['_date'].min()} to {train_df['_date'].max()}")
    print(
        "Validation period: "
        f"{validation_df['_date'].min()} to {validation_df['_date'].max()}"
    )
    print("Final 20% is not evaluated by this script.")
    print(f"Threshold for both models: {THRESHOLD}")

    results = []

    for name, features in FEATURE_SETS.items():
        results.append(
            evaluate_model(name, features, train_df, validation_df)
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 72)
    print("VALIDATION COMPARISON")
    print("=" * 72)
    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("Existing saved model and Flask application were not modified.")


if __name__ == "__main__":
    main()
