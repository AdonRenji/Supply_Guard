
from pathlib import Path
import time
import json

import pandas as pd
import joblib

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
    classification_report,
)

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "DataCoSupplyChainDataset.csv"
MODEL_DIR = BASE_DIR / "saved_model_25_features"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Late_delivery_risk"
DATE_COLUMN = "order date (DateOrders)"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.41

# --------------------------------------------------
# EXACT 25 FEATURES
# --------------------------------------------------

FEATURES = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
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
    "Type",
]

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Interaction_ShippingMode_OrderHour",
    "Type",
]


def create_features(df):
    """Create the same 25 input features used in the experiment."""
    df = df.copy()

    dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    df["Order Month"] = dates.dt.month
    df["Order Day"] = dates.dt.day
    df["Order Day of Week"] = dates.dt.dayofweek
    df["Order Hour"] = dates.dt.hour
    df["Order Week of Year"] = dates.dt.isocalendar().week.astype("float64")
    df["Order Day of Month"] = dates.dt.day

    df["Is Weekend"] = (dates.dt.dayofweek >= 5).astype("float64")
    df["Is Month Start"] = dates.dt.is_month_start.astype("float64")
    df["Is Month End"] = dates.dt.is_month_end.astype("float64")
    df["Is Quarter Start"] = dates.dt.is_quarter_start.astype("float64")
    df["Is Quarter End"] = dates.dt.is_quarter_end.astype("float64")

    df["Is Business Hour"] = (
        (dates.dt.hour >= 9) & (dates.dt.hour < 17)
    ).astype("float64")

    df["Is Night"] = (
        (dates.dt.hour < 6) | (dates.dt.hour >= 22)
    ).astype("float64")

    df["Interaction_ShippingMode_OrderHour"] = (
        df["Shipping Mode"].astype("string")
        + "_"
        + df["Order Hour"].astype("string")
    )

    return df


def main():
    print("=" * 65)
    print("SUPPLYGUARD - FINAL 25-FEATURE MODEL")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n"
            "Place DataCoSupplyChainDataset.csv beside this script."
        )

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    required = [TARGET, DATE_COLUMN, "Shipping Mode"] + FEATURES
    missing_raw = [
        col for col in required
        if col not in df.columns
        and col not in {
            "Order Month", "Order Day", "Order Day of Week",
            "Order Hour", "Order Week of Year", "Order Day of Month",
            "Is Weekend", "Is Month Start", "Is Month End",
            "Is Quarter Start", "Is Quarter End", "Is Business Hour",
            "Is Night", "Interaction_ShippingMode_OrderHour",
        }
    ]

    if missing_raw:
        raise ValueError(f"Missing required dataset columns: {missing_raw}")

    df = create_features(df)
    df = df.dropna(subset=[DATE_COLUMN, TARGET]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Usable rows: {len(df)}")
    print(f"Features: {len(FEATURES)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numerical_features = [
        col for col in FEATURES
        if col not in CATEGORICAL_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numerical_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

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

    print("\nTraining model...")
    start = time.time()
    model.fit(X_train, y_train)
    training_seconds = round(time.time() - start, 2)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "Model": "SupplyGuard 25-feature Random Forest",
        "Feature Count": len(FEATURES),
        "Train Rows": len(X_train),
        "Test Rows": len(X_test),
        "Random State": RANDOM_STATE,
        "Threshold": THRESHOLD,
        "Accuracy (%)": round(accuracy_score(y_test, predictions) * 100, 2),
        "Precision (%)": round(
            precision_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "Recall (%)": round(
            recall_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "F1-score (%)": round(
            f1_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "ROC-AUC (%)": round(
            roc_auc_score(y_test, probabilities) * 100, 2
        ),
        "Training Time (seconds)": training_seconds,
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "True Positives": int(cm[1, 1]),
    }

    # Save the complete fitted pipeline and its configuration.
    joblib.dump(model, MODEL_DIR / "supplyguard_25_model.joblib")

    (MODEL_DIR / "supplyguard_25_features.json").write_text(
        json.dumps(FEATURES, indent=2),
        encoding="utf-8",
    )

    (MODEL_DIR / "supplyguard_25_config.json").write_text(
        json.dumps({
            "target": TARGET,
            "date_column": DATE_COLUMN,
            "threshold": THRESHOLD,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "feature_count": len(FEATURES),
        }, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([metrics]).to_csv(
        MODEL_DIR / "supplyguard_25_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL RESULTS")
    print("=" * 65)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nSaved files in:")
    print(MODEL_DIR.resolve())
    print("\nExisting project and app.py were not modified.")


if __name__ == "__main__":
    main()
