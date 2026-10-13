

import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Paths and dataset
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "DataCoSupplyChainDataset.csv"

OUTPUT_DIR = (
    PROJECT_DIR / "results" / "feature_group_experiments"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH, encoding="latin1")

date_col = "order date (DateOrders)"
target_col = "Late_delivery_risk"
region_col = "Order Region"

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(
    subset=[date_col, target_col, region_col]
).copy()

df = df.sort_values(date_col).reset_index(drop=True)

df["Order Month"] = df[date_col].dt.month
df["Order Day of Week"] = df[date_col].dt.dayofweek
df["Order Hour"] = df[date_col].dt.hour
df[target_col] = df[target_col].astype(int)

# --------------------------------------------------
# 2. Same eight features and chronological split
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

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

X_train = train[features]
y_train = train[target_col]
X_test = test[features]
y_test = test[target_col]

print("=" * 70)
print("SUPPLYGUARD - CHRONOLOGICAL PREDICTION DIAGNOSTICS")
print("=" * 70)
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Test period:", test[date_col].min(), "to", test[date_col].max())

# --------------------------------------------------
# 3. Same preprocessing and Random Forest settings
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

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]),
        numeric_features,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

# --------------------------------------------------
# 4. Train diagnostic model and predict test orders
# --------------------------------------------------

print("\nTraining a separate diagnostic model...")
start = time.time()
model.fit(X_train, y_train)
print(f"Training completed in {time.time() - start:.2f} seconds.")

probabilities = model.predict_proba(X_test)[:, 1]
threshold = 0.41
predictions = (probabilities >= threshold).astype(int)

# --------------------------------------------------
# 5. Save row-level predictions
# --------------------------------------------------

diagnostics = pd.DataFrame({
    "Order Date": test[date_col].to_numpy(),
    "Order Region": test[region_col].to_numpy(),
    "Order Country": test["Order Country"].to_numpy(),
    "Actual Risk": y_test.to_numpy(),
    "Predicted Probability": probabilities,
    "Predicted Risk": predictions,
})

diagnostics["Correct Prediction"] = (
    diagnostics["Actual Risk"] == diagnostics["Predicted Risk"]
)

diagnostics["Error Type"] = "Correct"
diagnostics.loc[
    (diagnostics["Actual Risk"] == 0)
    & (diagnostics["Predicted Risk"] == 1),
    "Error Type",
] = "False Positive"

diagnostics.loc[
    (diagnostics["Actual Risk"] == 1)
    & (diagnostics["Predicted Risk"] == 0),
    "Error Type",
] = "False Negative"

predictions_path = OUTPUT_DIR / "chronological_test_predictions.csv"
diagnostics.to_csv(predictions_path, index=False)

# --------------------------------------------------
# 6. Summarize errors by region
# --------------------------------------------------

region_summary = diagnostics.groupby("Order Region").agg(
    Orders=("Actual Risk", "size"),
    Actual_Late_Risk_Rate=("Actual Risk", "mean"),
    Predicted_Late_Risk_Rate=("Predicted Risk", "mean"),
    Accuracy=("Correct Prediction", "mean"),
    False_Positives=("Error Type", lambda s: (s == "False Positive").sum()),
    False_Negatives=("Error Type", lambda s: (s == "False Negative").sum()),
).reset_index()

for col in [
    "Actual_Late_Risk_Rate",
    "Predicted_Late_Risk_Rate",
    "Accuracy",
]:
    region_summary[col] = (region_summary[col] * 100).round(2)

region_summary = region_summary.sort_values(
    "Orders", ascending=False
)

region_path = OUTPUT_DIR / "chronological_region_diagnostics.csv"
region_summary.to_csv(region_path, index=False)

# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

print("\nOverall diagnostic results:")
print("Threshold:", threshold)
print(f"Accuracy: {diagnostics['Correct Prediction'].mean() * 100:.2f}%")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities) * 100:.2f}%")

print("\nPrediction counts:")
print(diagnostics["Error Type"].value_counts().to_string())

print("\nRegions with the most test orders:")
print(region_summary.head(15).to_string(index=False))

print("\nFiles saved:")
print(predictions_path)
print(region_path)

print("\nThe existing final model and Flask app were not modified.")
print("=" * 70)
