
import pandas as pd
from pathlib import Path

# Load the dataset from the project folder
project_folder = Path(__file__).resolve().parent
dataset_path = project_folder / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(dataset_path, encoding="latin1")

print("\n--- Target column ---")
print(df["Late_delivery_risk"].value_counts(dropna=False))
print("\nTarget data type:", df["Late_delivery_risk"].dtype)

print("\n--- Related delivery columns ---")
for col in df.columns:
    if any(word in col.lower() for word in [
        "late_delivery_risk",
        "late_delivery",
        "delivery_status",
        "days for shipping (real)",
        "days for shipment (scheduled)",
        "actual shipping"
    ]):
        print(f"\nColumn: {col}")
        print(df[col].value_counts(dropna=False).head(10))

print("\n--- Sample rows ---")
columns_to_check = [
    col for col in [
        "Late_delivery_risk",
        "Delivery Status",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "order date (DateOrders)",
        "shipping date (DateOrders)",
        "Order Status"
    ]
    if col in df.columns
]

print(df[columns_to_check].head(10).to_string(index=False))

