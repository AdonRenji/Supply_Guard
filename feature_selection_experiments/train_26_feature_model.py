
from pathlib import Path
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
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = BASE_DIR / "results" / "feature_group_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "25_feature_random_split_metrics.csv"

TARGET = "Late_delivery_risk"
DATE_COLUMN = "order date (DateOrders)"
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.41

# Original 31 features minus six selected features.
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


def main():
    print("=" * 65)
    print("SUPPLYGUARD: 25-FEATURE RANDOM-SPLIT EXPERIMENT")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN, TARGET]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()

    # Recreate date-derived features.
    dates = df[DATE_COLUMN]
    df["Order Month"] = dates.dt.month
    df["Order Day"] = dates.dt.day
    df["Order Day of Week"] = dates.dt.dayofweek
    df["Order Hour"] = dates.dt.hour
    df["Order Week of Year"] = dates.dt.isocalendar().week.astype(int)
    df["Order Day of Month"] = dates.dt.day
    df["Is Weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    df["Is Month Start"] = dates.dt.is_month_start.astype(int)
    df["Is Month End"] = dates.dt.is_month_end.astype(int)
    df["Is Quarter Start"] = dates.dt.is_quarter_start.astype(int)
    df["Is Quarter End"] = dates.dt.is_quarter_end.astype(int)
    df["Is Business Hour"] = (
        (dates.dt.hour >= 9) & (dates.dt.hour < 17)
    ).astype(int)
    df["Is Night"] = (
        (dates.dt.hour < 6) | (dates.dt.hour >= 22)
    ).astype(int)
    df["Interaction_ShippingMode_OrderHour"] = (
        df["Shipping Mode"].astype(str)
        + "_"
        + df["Order Hour"].astype(str)
    )

    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    print(f"Usable rows: {len(df)}")
    print(f"Feature count: {len(FEATURES)}")
    print("\nFeatures used:")
    for feature in FEATURES:
        print(f" - {feature}")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Random stratified 80/20 split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    categorical = [
        f for f in FEATURES if f in CATEGORICAL_FEATURES
    ]
    numerical = [
        f for f in FEATURES if f not in CATEGORICAL_FEATURES
    ]

    transformers = [
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numerical,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ]

    model = Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("\nTraining only the 25-feature model...")
    start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    results = {
        "Model": "25-feature model",
        "Feature Count": len(FEATURES),
        "Train Rows": len(X_train),
        "Test Rows": len(X_test),
        "Random State": RANDOM_STATE,
        "Threshold": THRESHOLD,
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
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
        "Training Time (seconds)": round(training_seconds, 2),
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "True Positives": int(cm[1, 1]),
    }

    pd.DataFrame([results]).to_csv(RESULT_PATH, index=False)

    print("\n" + "=" * 65)
    print("25-FEATURE MODEL RESULTS")
    print("=" * 65)
    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(cm)
    print(f"\nResults saved to: {RESULT_PATH}")
    print("The existing saved model and app.py were not modified.")


if __name__ == "__main__":
    main()
