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

THRESHOLD_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_threshold.txt"
)

DATASET_PATH = "DataCoSupplyChainDataset.csv"

OUTPUT_PATH = "test_records_10.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_PATH)

if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read().strip())
else:
    threshold = 0.41

print(f"Decision threshold: {threshold}")


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

print("\nRunning predictions on TEST SET...")
print("Please wait...")

test_probability = model.predict_proba(X_test)[:, 1]

test_prediction = (
    test_probability >= threshold
).astype(int)


# ============================================================
# CREATE TEST RESULTS
# ============================================================

test_results = df.loc[X_test.index].copy()

test_results["Actual"] = y_test
test_results["Probability"] = test_probability
test_results["Prediction"] = test_prediction


# ============================================================
# CORRECTLY PREDICTED LATE-RISK
# ============================================================

correct_late_risk = test_results[
    (test_results["Actual"] == 1)
    & (test_results["Prediction"] == 1)
].copy()

correct_late_risk = correct_late_risk.sort_values(
    "Probability",
    ascending=False
)

late_risk_5 = correct_late_risk.head(5)


# ============================================================
# CORRECTLY PREDICTED NO-RISK
# ============================================================

correct_no_risk = test_results[
    (test_results["Actual"] == 0)
    & (test_results["Prediction"] == 0)
].copy()

correct_no_risk = correct_no_risk.sort_values(
    "Probability"
)

no_risk_5 = correct_no_risk.head(5)


# ============================================================
# ADD GROUP HEADINGS
# ============================================================

late_risk_5 = late_risk_5.copy()
no_risk_5 = no_risk_5.copy()

late_risk_5["Prediction Type"] = "LATE DELIVERY RISK"
no_risk_5["Prediction Type"] = "NO RISK"


# ============================================================
# COMBINE 5 + 5
# ============================================================

selected = pd.concat(
    [
        late_risk_5,
        no_risk_5
    ]
)


# ============================================================
# CREATE CSV
# ============================================================

output = pd.DataFrame()

# Group heading
output["Prediction Type"] = selected["Prediction Type"]


# Shipment Information
output["Shipping Mode"] = selected["Shipping Mode"]

output["Days for Shipment (Scheduled)"] = (
    selected["Days for shipment (scheduled)"]
)

output["Market"] = selected["Market"]


# Customer Information
output["Customer Country"] = selected["Customer Country"]
output["Customer State"] = selected["Customer State"]
output["Customer City"] = selected["Customer City"]
output["Customer Segment"] = selected["Customer Segment"]

output["Order Region"] = selected["Order Region"]
output["Order Country"] = selected["Order Country"]
output["Order State"] = selected["Order State"]


# Product Information
output["Category Name"] = selected["Category Name"]
output["Department Name"] = selected["Department Name"]
output["Type"] = selected["Type"]

output["Order Item Quantity"] = (
    selected["Order Item Quantity"]
)

output["Order Item Product Price"] = (
    selected["Order Item Product Price"]
)

output["Order Item Discount Rate"] = (
    selected["Order Item Discount Rate"]
)

output["Sales"] = selected["Sales"]


# Order Information
output["Order Date"] = (
    order_datetime.loc[selected.index]
    .dt.strftime("%Y/%m/%d")
)

output["Order Time"] = (
    order_datetime.loc[selected.index]
    .dt.strftime("%H:%M")
)


# ============================================================
# VERIFICATION INFORMATION
# ============================================================

output["Actual Late_delivery_risk"] = (
    selected["Actual"]
)

output["Model Probability"] = (
    selected["Probability"] * 100
).round(2)

output["Decision Threshold"] = threshold

output["Model Prediction"] = (
    selected["Prediction"]
)

output["Dataset Split"] = (
    "Unseen 20% Test Set"
)


# ============================================================
# SAVE CSV
# ============================================================

output.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("TEST RECORD CSV CREATED")
print("=" * 70)

print(f"\nFile: {OUTPUT_PATH}")
print(f"Total records: {len(output)}")

print("\nLATE DELIVERY RISK : 5")
print("NO RISK             : 5")

print("\nAll records are from the UNSEEN 20% TEST SET.")

print("\nSelected records:")

print(
    output[
        [
            "Prediction Type",
            "Actual Late_delivery_risk",
            "Model Probability",
            "Model Prediction"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)