import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = "DataCoSupplyChainDataset.csv"

MODEL_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_model.joblib"
)

THRESHOLD_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_threshold.txt"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("SUPPLYGUARD - TEST PREDICTION VERIFICATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    encoding="latin1"
)

print(f"Dataset rows: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")


# ============================================================
# 3. CREATE DATE FEATURES
# ============================================================

print("\nCreating date features...")

date_column = "order date (DateOrders)"

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

df["Order Month"] = df[date_column].dt.month
df["Order Day"] = df[date_column].dt.day
df["Order Day of Week"] = df[date_column].dt.dayofweek
df["Order Hour"] = df[date_column].dt.hour

df["Is Weekend"] = (
    df["Order Day of Week"] >= 5
).astype(int)

df["Order Week of Year"] = (
    df[date_column].dt.isocalendar().week.astype(int)
)

df["Order Day of Month"] = df[date_column].dt.day

df["Is Month Start"] = (
    df[date_column].dt.is_month_start
).astype(int)

df["Is Month End"] = (
    df[date_column].dt.is_month_end
).astype(int)

df["Is Quarter Start"] = (
    df[date_column].dt.is_quarter_start
).astype(int)

df["Is Quarter End"] = (
    df[date_column].dt.is_quarter_end
).astype(int)

df["Is Business Hour"] = (
    (df["Order Hour"] >= 9) &
    (df["Order Hour"] <= 17) &
    (df["Order Day of Week"] < 5)
).astype(int)

df["Is Night"] = (
    (df["Order Hour"] < 6) |
    (df["Order Hour"] >= 22)
).astype(int)


# ============================================================
# 4. CREATE INTERACTION FEATURE
# ============================================================

print("Creating interaction feature...")

df["Interaction_ShippingMode_OrderHour"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Order Hour"].astype(str)
)


# ============================================================
# 5. FINAL 31 FEATURES
# ============================================================

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
    "Customer State"
]

print(f"\nNumber of features: {len(features)}")


# ============================================================
# 6. TARGET
# ============================================================

target = "Late_delivery_risk"


# ============================================================
# 7. PREPARE X AND y
# ============================================================

X = df[features].copy()
y = df[target].copy()


# ============================================================
# 8. RECREATE SAME 80/20 TEST SPLIT
# ============================================================

print("\nCreating the same 80/20 stratified split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ============================================================
# 9. LOAD SAVED MODEL
# ============================================================

print("\nLoading saved final model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Saved model not found!")
    print(
        "Make sure train_final_model.py has been completed."
    )
    exit()

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# 10. LOAD THRESHOLD
# ============================================================

if os.path.exists(THRESHOLD_PATH):

    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read().strip())

else:

    threshold = 0.41


# ============================================================
# 11. SELECT ONE UNSEEN TEST RECORD
# ============================================================

# We select the first record from the test set.
# This record was NOT used for training.

test_record = X_test.iloc[[0]]
actual_value = int(y_test.iloc[0])


# ============================================================
# 12. PREDICT
# ============================================================

probability = model.predict_proba(
    test_record
)[0][1]

prediction = int(
    probability >= threshold
)


# ============================================================
# 13. DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 60)
print("TEST RECORD VERIFICATION")
print("=" * 60)

print("\nActual Late_delivery_risk:")
print(actual_value)

print("\nPredicted probability of late delivery:")
print(f"{probability * 100:.2f}%")

print("\nClassification threshold:")
print(f"{threshold:.2f}")

print("\nPredicted Late_delivery_risk:")
print(prediction)


# ============================================================
# 14. CHECK WHETHER PREDICTION IS CORRECT
# ============================================================

if prediction == actual_value:

    print("\nRESULT: CORRECT PREDICTION")

else:

    print("\nRESULT: INCORRECT PREDICTION")


# ============================================================
# 15. SHOW INTERPRETATION
# ============================================================

print("\nInterpretation:")

if prediction == 1:

    print("Model predicts HIGH RISK of late delivery.")

else:

    print("Model predicts LOW RISK / likely on-time delivery.")


# ============================================================
# 16. SHOW SOME INPUT FEATURES
# ============================================================

print("\nSelected test-record details:")
print("-" * 60)

for feature in [
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
    "Order Item Product Price",
    "Sales",
    "Order Month",
    "Order Day",
    "Order Hour",
    "Customer City",
    "Customer State",
    "Type"
]:

    print(f"{feature}: {test_record.iloc[0][feature]}")


print("\n")
print("=" * 60)
print("VERIFICATION COMPLETED")
print("=" * 60)