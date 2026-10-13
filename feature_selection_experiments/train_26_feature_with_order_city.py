
import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "DataCoSupplyChainDataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "feature_group_experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.41
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH, encoding="latin1")

target = "Late_delivery_risk"
date_col = "order date (DateOrders)"

dates = pd.to_datetime(df[date_col], errors="coerce")

df["Order Month"] = dates.dt.month
df["Order Day"] = dates.dt.day
df["Order Day of Week"] = dates.dt.dayofweek
df["Order Week of Year"] = dates.dt.isocalendar().week.astype(float)
df["Order Day of Month"] = dates.dt.day
df["Order Hour"] = dates.dt.hour
df["Is Weekend"] = (dates.dt.dayofweek >= 5).astype(float)
df["Is Month Start"] = dates.dt.is_month_start.astype(float)
df["Is Month End"] = dates.dt.is_month_end.astype(float)
df["Is Quarter Start"] = dates.dt.is_quarter_start.astype(float)
df["Is Quarter End"] = dates.dt.is_quarter_end.astype(float)
df["Is Business Hour"] = (
    (df["Order Hour"] >= 9) & (df["Order Hour"] < 17)
).astype(float)
df["Is Night"] = (
    (df["Order Hour"] < 6) | (df["Order Hour"] >= 22)
).astype(float)

df["Interaction_ShippingMode_OrderHour"] = (
    df["Shipping Mode"].astype(str) + "_" + df["Order Hour"].astype(str)
)

features = [
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
    "Order City",
]

missing = [c for c in features + [target] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

data = df[features + [target]].copy()
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target])
data = data[data[target].isin([0, 1])]
data[target] = data[target].astype(int)

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20,
    random_state=RANDOM_STATE, stratify=y
)

categorical = [
    "Shipping Mode", "Market", "Order Region", "Order Country",
    "Order State", "Interaction_ShippingMode_OrderHour",
    "Type", "Order City"
]
numeric = [f for f in features if f not in categorical]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

print("Training 26-feature model with Order City...")
print(f"Rows: {len(data)} | Features: {len(features)}")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

results = {
    "Model": "26-feature model (+ Order City)",
    "Feature Count": len(features),
    "Random State": RANDOM_STATE,
    "Threshold": THRESHOLD,
    "Accuracy (%)": round(accuracy_score(y_test, predictions) * 100, 2),
    "Precision (%)": round(precision_score(y_test, predictions, zero_division=0) * 100, 2),
    "Recall (%)": round(recall_score(y_test, predictions, zero_division=0) * 100, 2),
    "F1-score (%)": round(f1_score(y_test, predictions, zero_division=0) * 100, 2),
    "ROC-AUC (%)": round(roc_auc_score(y_test, probabilities) * 100, 2),
    "True Negatives": int(cm[0, 0]),
    "False Positives": int(cm[0, 1]),
    "False Negatives": int(cm[1, 0]),
    "True Positives": int(cm[1, 1]),
    "Training Time (seconds)": round(training_time, 2),
}

print("\n26-FEATURE RESULTS")
for key, value in results.items():
    print(f"{key}: {value}")
print("\nConfusion matrix:")
print(cm)

output_path = os.path.join(OUTPUT_DIR, "26_feature_order_city_metrics.csv")
pd.DataFrame([results]).to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print("Existing saved model and app.py were not modified.")
