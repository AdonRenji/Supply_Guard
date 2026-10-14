from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

# This script and the dataset are in the same folder.
PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "DataCoSupplyChainDataset.csv"
MODEL_DIR = PROJECT_DIR / "saved_model"
RESULTS_DIR = PROJECT_DIR / "results"

DATE_COLUMN = "order date (DateOrders)"
TARGET = "Late_delivery_risk"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200
THRESHOLDS = [0.41, 0.50]


# ==================================================
# 2. FINALIZED 19 FEATURES
# ==================================================

FEATURES_19 = [
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
    "Type",
    "Order Month",
    "Order Day",
    "Order Day of Week",
    "Order Hour",
    "Is Weekend",
    "Order Week of Year",
    "Is Night",
    "Interaction_ShippingMode_OrderHour",
]

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Type",
    "Interaction_ShippingMode_OrderHour",
]

NUMERIC_FEATURES = [
    feature
    for feature in FEATURES_19
    if feature not in CATEGORICAL_FEATURES
]


# ==================================================
# 3. DATE/TIME FEATURE ENGINEERING
# ==================================================

def create_features(df):
    df = df.copy()

    dates = pd.to_datetime(
        df[DATE_COLUMN], errors="coerce"
    )

    df["Order Month"] = dates.dt.month
    df["Order Day"] = dates.dt.day
    df["Order Day of Week"] = dates.dt.dayofweek
    df["Order Hour"] = dates.dt.hour

    df["Is Weekend"] = (
        dates.dt.dayofweek.isin([5, 6]).astype("int8")
    )

    df["Order Week of Year"] = (
        dates.dt.isocalendar().week.astype("float")
    )

    # Night means before 06:00 or from 22:00 onward.
    df["Is Night"] = (
        (dates.dt.hour < 6) |
        (dates.dt.hour >= 22)
    ).astype("int8")

    # Shipping mode and order hour interaction.
    df["Interaction_ShippingMode_OrderHour"] = (
        df["Shipping Mode"].fillna("Missing").astype(str)
        + "_"
        + df["Order Hour"].fillna(-1).astype(int).astype(str)
    )

    return df


# ==================================================
# 4. LOAD AND VALIDATE DATASET
# ==================================================

def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place DataCoSupplyChainDataset.csv "
            "beside train_model.py."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SupplyGuard dataset...")

    df = pd.read_csv(
        DATASET_PATH,
        encoding="cp1252",
        low_memory=False,
    )

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Missing date column: {DATE_COLUMN}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    df = create_features(df)

    missing = [
        feature for feature in FEATURES_19
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    data = df[FEATURES_19 + [TARGET]].copy()

    data[TARGET] = pd.to_numeric(
        data[TARGET], errors="coerce"
    )

    data = data.dropna(subset=[TARGET])

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    X = data[FEATURES_19]
    y = data[TARGET].astype(int)

    print("Dataset rows:", len(data))
    print("Selected features:", len(FEATURES_19))
    print("Target:", TARGET)


    # ==================================================
    # 5. SPLIT DATA
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))


    # ==================================================
    # 6. PREPROCESSING
    # ==================================================

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        )),
    ])

    preprocessing = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])


    # ==================================================
    # 7. RANDOM FOREST MODEL
    # ==================================================

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("\nTraining 19-feature Random Forest...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_seconds = time.time() - start_time

    print(
        f"Training completed in "
        f"{training_seconds:.2f} seconds."
    )


    # ==================================================
    # 8. EVALUATION
    # ==================================================

    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test, predictions
        )

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        result = {
            "features": len(FEATURES_19),
            "trees": N_ESTIMATORS,
            "threshold": threshold,
            "accuracy_percent": round(
                accuracy * 100, 3
            ),
            "precision_percent": round(
                precision * 100, 3
            ),
            "recall_percent": round(
                recall * 100, 3
            ),
            "f1_percent": round(
                f1 * 100, 3
            ),
            "roc_auc_percent": round(
                auc * 100, 3
            ),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "training_seconds": round(
                training_seconds, 2
            ),
        }

        results.append(result)

        print("\n" + "=" * 45)
        print(f"THRESHOLD: {threshold}")
        print("=" * 45)
        print(f"Accuracy:  {accuracy * 100:.3f}%")
        print(f"Precision: {precision * 100:.3f}%")
        print(f"Recall:    {recall * 100:.3f}%")
        print(f"F1-score:  {f1 * 100:.3f}%")
        print(f"ROC-AUC:   {auc * 100:.3f}%")
        print(
            f"TN: {tn} | FP: {fp} | "
            f"FN: {fn} | TP: {tp}"
        )


    # ==================================================
    # 9. SAVE TRAINED MODEL AND CONFIGURATION
    # ==================================================

    model_path = MODEL_DIR / "supplyguard_19_model.joblib"

    joblib.dump(model, model_path)

    config = {
        "features": FEATURES_19,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "thresholds": THRESHOLDS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "training_seconds": round(training_seconds, 2),
    }

    with open(
        MODEL_DIR / "model_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)


    # ==================================================
    # 10. SAVE EVALUATION RESULTS
    # ==================================================

    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "metrics.csv",
        index=False,
    )

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 45)
    print("SUPPLYGUARD 19-FEATURE MODEL COMPLETE")
    print("=" * 45)
    print("Model saved:", model_path)
    print("Configuration saved:", MODEL_DIR / "model_config.json")
    print("Metrics saved:", RESULTS_DIR / "metrics.csv")
    print("Metrics saved:", RESULTS_DIR / "metrics.json")
    print("Original project files were not modified.")


if __name__ == "__main__":
    main()