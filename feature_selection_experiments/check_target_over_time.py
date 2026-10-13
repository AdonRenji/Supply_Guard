
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATASET_PATH, encoding="latin1")

date_col = "order date (DateOrders)"
target_col = "Late_delivery_risk"

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, target_col]).copy()

df[target_col] = df[target_col].astype(int)
df["Order Year-Month"] = df[date_col].dt.to_period("M").astype(str)

print("=" * 70)
print("SUPPLYGUARD - TARGET DISTRIBUTION OVER TIME")
print("=" * 70)

print("\nDataset date range:")
print("Start:", df[date_col].min())
print("End:  ", df[date_col].max())
print("Rows:", len(df))

print("\nOverall target counts:")
print(df[target_col].value_counts().sort_index())

print("\nOverall target percentages:")
print((df[target_col].value_counts(normalize=True).sort_index() * 100).round(2))

monthly = df.groupby("Order Year-Month").agg(
    Orders=(target_col, "size"),
    Late_Risk_Count=(target_col, "sum"),
    Late_Risk_Rate=(target_col, "mean"),
)

monthly["Late_Risk_Rate"] = (
    monthly["Late_Risk_Rate"] * 100
).round(2)

print("\nMonthly target distribution:")
print(monthly.to_string())

print("\n--- Target by calendar year ---")
yearly = df.groupby(df[date_col].dt.year).agg(
    Orders=(target_col, "size"),
    Late_Risk_Count=(target_col, "sum"),
    Late_Risk_Rate=(target_col, "mean"),
)

yearly["Late_Risk_Rate"] = (
    yearly["Late_Risk_Rate"] * 100
).round(2)

print(yearly.to_string())

print("\nAnalysis complete. No model was trained or modified.")

