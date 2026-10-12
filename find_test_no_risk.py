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
# SAME 80/20 STRATIFIED SPLIT
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
# PREDICT ONLY ON UNSEEN TEST SET
# ============================================================

print("\nPredicting on unseen TEST SET...")
print("Please wait...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= 0.41).astype(int)


# ============================================================
# CREATE TEST RESULT DATAFRAME
# ============================================================

test_results = df.loc[X_test.index].copy()

test_results["Actual"] = y_test
test_results["Probability"] = probabilities
test_results["Prediction"] = predictions


# ============================================================
# FIND CORRECTLY PREDICTED NO-RISK RECORDS
# ============================================================

correct_no_risk = test_results[
    (test_results["Actual"] == 0)
    & (test_results["Prediction"] == 0)
].copy()


# Sort from lowest probability to highest
correct_no_risk = correct_no_risk.sort_values(
    "Probability",
    ascending=True
)


# Take 3 records
selected_records = correct_no_risk.head(3)


# ============================================================
# PRINT 3 RECORDS
# ============================================================

print("\n")
print("=" * 60)
print("3 CORRECTLY PREDICTED NO-RISK RECORDS")
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

    print(
        f"Shipping Mode              : "
        f"{row['Shipping Mode']}"
    )

    print(
        f"Days for Shipment          : "
        f"{row['Days for shipment (scheduled)']}"
    )

    print(
        f"Customer Country           : "
        f"{row['Customer Country']}"
    )

    print(
        f"Category Name              : "
        f"{row['Category Name']}"
    )

    print(
        f"Order Item Quantity        : "
        f"{row['Order Item Quantity']}"
    )

    print(
        f"Market                     : "
        f"{row['Market']}"
    )

    print(
        f"Order Region               : "
        f"{row['Order Region']}"
    )

    print(
        f"Order Country              : "
        f"{row['Order Country']}"
    )

    print(
        f"Order State                : "
        f"{row['Order State']}"
    )

    print(
        f"Customer Segment           : "
        f"{row['Customer Segment']}"
    )

    print(
        f"Department Name            : "
        f"{row['Department Name']}"
    )

    print(
        f"Product Price              : "
        f"{row['Order Item Product Price']}"
    )

    print(
        f"Discount Rate              : "
        f"{row['Order Item Discount Rate']}"
    )

    print(
        f"Sales                      : "
        f"{row['Sales']}"
    )

    print(
        f"Customer City              : "
        f"{row['Customer City']}"
    )

    print(
        f"Customer State             : "
        f"{row['Customer State']}"
    )

    print(
        f"Type                       : "
        f"{row['Type']}"
    )

    print(
        f"Order Date                 : "
        f"{original_datetime.strftime('%Y/%m/%d')}"
    )

    print(
        f"Order Time                 : "
        f"{original_datetime.strftime('%H:%M')}"
    )

    print("--------------------------------------------")

    print(
        f"Actual Late_delivery_risk  : "
        f"{int(row['Actual'])}"
    )

    print(
        f"Model Probability          : "
        f"{row['Probability'] * 100:.2f}%"
    )

    print(
        f"Decision Threshold         : "
        f"{0.41}"
    )

    print(
        f"Model Prediction           : "
        f"{int(row['Prediction'])}"
    )

    print(
        f"Dataset Split              : "
        f"UNSEEN 20% TEST SET"
    )

    print("--------------------------------------------")


print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)
print("These 3 records are from the unseen 20% test set.")
print("All 3 are correctly predicted as NO RISK.")
print("=" * 60)