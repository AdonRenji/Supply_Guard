
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
dataset_path = project_folder / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(dataset_path, encoding="latin1")

date_column = "order date (DateOrders)"

# Create date-derived features in memory
df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

df["Order Month"] = df[date_column].dt.month
df["Order Day of Week"] = df[date_column].dt.dayofweek
df["Order Hour"] = df[date_column].dt.hour

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

print("=" * 65)
print("SUPPLYGUARD - FEATURE AVAILABILITY AUDIT")
print("=" * 65)

for feature in features:
    print(f"\nFeature: {feature}")
    print("Missing values:", df[feature].isna().sum())
    print("Distinct values:", df[feature].nunique(dropna=True))

print("\n--- Date and shipping fields ---")

for column in [
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Days for shipment (scheduled)",
    "Days for shipping (real)",
    "Delivery Status",
    "Late_delivery_risk",
]:
    print(f"{column}: {'Present' if column in df.columns else 'Not found'}")

print("\n--- Sample timestamps and scheduled days ---")
columns = [
    "order date (DateOrders)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
]
print(df[columns].head(10).to_string(index=False))

print("\nNo model was trained or modified.")

