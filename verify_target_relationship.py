
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
dataset_path = project_folder / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(dataset_path, encoding="latin1")

print("\n--- Delivery Status vs Late_delivery_risk ---")
print(pd.crosstab(
    df["Delivery Status"],
    df["Late_delivery_risk"],
    margins=True
))

print("\n--- Late delivery status check ---")
late_rows = df[df["Delivery Status"] == "Late delivery"]

print("Rows marked Late delivery:", len(late_rows))
print(
    "Late_delivery_risk = 1:",
    (late_rows["Late_delivery_risk"] == 1).sum()
)
print(
    "Late_delivery_risk = 0:",
    (late_rows["Late_delivery_risk"] == 0).sum()
)

print("\n--- Compare actual and scheduled shipping days ---")
df["Actual_exceeds_schedule"] = (
    df["Days for shipping (real)"]
    > df["Days for shipment (scheduled)"]
)

print(pd.crosstab(
    df["Actual_exceeds_schedule"],
    df["Late_delivery_risk"],
    margins=True
))

