import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_model.joblib"
)

DATASET_PATH = "DataCoSupplyChainDataset.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")
model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    encoding="latin1"
)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# CREATE TEMPORAL FEATURES
# ============================================================

print("Creating temporal features...")

order_datetime = pd.to_datetime(
    df["order date (DateOrders)"],
    format="%m/%d/%Y %H:%M",
    errors="coerce"
)

df["Order Month"] = order_datetime.dt.month
df["Order Day"] = order_datetime.dt.day
df["Order Day of Week"] = order_datetime.dt.dayofweek
df["Order Hour"] = order_datetime.dt.hour

df["Is Weekend"] = (
    df["Order Day of Week"] >= 5
).astype(int)

df["Order Week of Year"] = (
    order_datetime.dt.isocalendar().week.astype(int)
)

df["Order Day of Month"] = order_datetime.dt.day

df["Is Month Start"] = (
    order_datetime.dt.is_month_start
).astype(int)

df["Is Month End"] = (
    order_datetime.dt.is_month_end
).astype(int)

df["Is Quarter Start"] = (
    order_datetime.dt.is_quarter_start
).astype(int)

df["Is Quarter End"] = (
    order_datetime.dt.is_quarter_end
).astype(int)

df["Is Business Hour"] = (
    (order_datetime.dt.dayofweek < 5)
    & (order_datetime.dt.hour >= 9)
    & (order_datetime.dt.hour < 17)
).astype(int)

df["Is Night"] = (
    (order_datetime.dt.hour < 6)
    | (order_datetime.dt.hour >= 22)
).astype(int)


# ============================================================
# INTERACTION FEATURE
# ============================================================

df["Interaction_ShippingMode_OrderHour"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Order Hour"].astype(str)
)


# ============================================================
# FINAL 31 FEATURES
# ============================================================

final_features = [
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


# ============================================================
# X AND Y
# ============================================================

X = df[final_features]
y = df["Late_delivery_risk"]


# ============================================================
# SAME 80/20 TEST SPLIT
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
print(f"Testing records : {len(X_test)}")


# ============================================================
# PREDICT TEST SET
# ============================================================

print("\nPredicting on unseen TEST SET...")
print("Please wait...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= 0.41
).astype(int)


# ============================================================
# CREATE RESULTS
# ============================================================

test_results = df.loc[X_test.index].copy()

test_results["Actual"] = y_test
test_results["Probability"] = probabilities
test_results["Prediction"] = predictions


# ============================================================
# CORRECTLY PREDICTED RISK ONLY
# ============================================================

correct_risk = test_results[
    (test_results["Actual"] == 1)
    & (test_results["Prediction"] == 1)
].copy()


# Highest probability first
correct_risk = correct_risk.sort_values(
    "Probability",
    ascending=False
)


# Select 3
selected_records = correct_risk.head(3)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 60)
print("3 CORRECTLY PREDICTED LATE-RISK RECORDS")
print("FROM THE UNSEEN 20% TEST SET")
print("=" * 60)


for i, (index, row) in enumerate(
    selected_records.iterrows(),
    start=1
):

    original_datetime = order_datetime.loc[index]

    print("\n")
    print(f"RECORD {i}")
    print("-" * 44)

    print("VALUES TO ENTER IN STREAMLIT")
    print("--------------------------------------------")

    print(f"Shipping Mode              : {row['Shipping Mode']}")
    print(f"Days for Shipment          : {row['Days for shipment (scheduled)']}")
    print(f"Customer Country           : {row['Customer Country']}")
    print(f"Category Name              : {row['Category Name']}")
    print(f"Order Item Quantity        : {row['Order Item Quantity']}")
    print(f"Market                     : {row['Market']}")
    print(f"Order Region               : {row['Order Region']}")
    print(f"Order Country              : {row['Order Country']}")
    print(f"Order State                : {row['Order State']}")
    print(f"Customer Segment           : {row['Customer Segment']}")
    print(f"Department Name            : {row['Department Name']}")
    print(f"Product Price              : {row['Order Item Product Price']}")
    print(f"Discount Rate              : {row['Order Item Discount Rate']}")
    print(f"Sales                      : {row['Sales']}")
    print(f"Customer City              : {row['Customer City']}")
    print(f"Customer State             : {row['Customer State']}")
    print(f"Type                       : {row['Type']}")
    print(f"Order Date                 : {original_datetime.strftime('%Y/%m/%d')}")
    print(f"Order Time                 : {original_datetime.strftime('%H:%M')}")

    print("--------------------------------------------")

    print(f"Actual Late_delivery_risk  : {int(row['Actual'])}")
    print(f"Model Probability          : {row['Probability'] * 100:.2f}%")
    print(f"Decision Threshold         : 0.41")
    print(f"Model Prediction           : {int(row['Prediction'])}")
    print(f"Dataset Split              : UNSEEN 20% TEST SET")

    print("--------------------------------------------")


print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)
print("3 Risk records displayed successfully.")
print("All records are from the unseen 20% test set.")
print("=" * 60)