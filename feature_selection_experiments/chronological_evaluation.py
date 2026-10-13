
import time
from pathlib import Path

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "feature_group_experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH, encoding="latin1")

print("=" * 70)
print("SUPPLYGUARD - CHRONOLOGICAL EVALUATION")
print("=" * 70)
print("Rows:", len(df))

# --------------------------------------------------
# 2. Prepare order date and target
# --------------------------------------------------

date_column = "order date (DateOrders)"
target_column = "Late_delivery_risk"

df[date_column] = pd.to_datetime(
    df[date_column], errors="coerce"
)

df = df.dropna(subset=[date_column, target_column]).copy()

df["Order Month"] = df[date_column].dt.month
df["Order Day of Week"] = df[date_column].dt.dayofweek
df["Order Hour"] = df[date_column].dt.hour

df[target_column] = df[target_column].astype(int)

# --------------------------------------------------
# 3. Select operational features
# --------------------------------------------------

features = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Order Country",
    "Order State",
    "Order Region",
    "Order Month",
    "Order Day of Week",
    "Order Hour",
]

X = df[features]
y = df[target_column]

# --------------------------------------------------
# 4. Split chronologically
#    Earlier 80% = training
#    Later 20% = testing
# --------------------------------------------------

df = df.sort_values(date_column).reset_index(drop=True)

X = df[features]
y = df[target_column]

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining period:")
print(df[date_column].iloc[0], "to",
      df[date_column].iloc[split_index - 1])

print("\nTesting period:")
print(df[date_column].iloc[split_index], "to",
      df[date_column].iloc[-1])

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).sort_index().round(4))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).sort_index().round(4))

# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

categorical_features = [
    "Shipping Mode",
    "Order Country",
    "Order State",
    "Order Region",
]

numeric_features = [
    "Days for shipment (scheduled)",
    "Order Month",
    "Order Day of Week",
    "Order Hour",
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# --------------------------------------------------
# 6. Build and train model
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

print("\nTraining chronological model...")
start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

# --------------------------------------------------
# 7. Evaluate at threshold 0.41
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]
threshold = 0.41
predictions = (probabilities >= threshold).astype(int)

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

print("\nConfusion matrix:")
print(cm)

metrics = {
    "Model": "Eight Operational Features - Chronological",
    "Feature Count": len(features),
    "Accuracy (%)": accuracy_score(y_test, predictions) * 100,
    "Precision (%)": precision_score(
        y_test, predictions, zero_division=0
    ) * 100,
    "Recall (%)": recall_score(
        y_test, predictions, zero_division=0
    ) * 100,
    "F1 Score (%)": f1_score(
        y_test, predictions, zero_division=0
    ) * 100,
    "ROC-AUC (%)": roc_auc_score(y_test, probabilities) * 100,
    "Threshold": threshold,
    "Training Time (seconds)": round(training_time, 2),
    "True Negatives": int(cm[0, 0]),
    "False Positives": int(cm[0, 1]),
    "False Negatives": int(cm[1, 0]),
    "True Positives": int(cm[1, 1]),
}

print("\nResults:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# --------------------------------------------------
# 8. Save results only
# --------------------------------------------------

output_path = OUTPUT_DIR / "chronological_evaluation.csv"

pd.DataFrame([metrics]).to_csv(output_path, index=False)

print("\nResults saved to:")
print(output_path)

print("\nExisting saved model and Flask app were not modified.")
print("=" * 70)

