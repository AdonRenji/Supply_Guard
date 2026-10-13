
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATASET_PATH, encoding="latin1")

date_col = "order date (DateOrders)"
target_col = "Late_delivery_risk"
region_col = "Order Region"

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, target_col, region_col]).copy()
df = df.sort_values(date_col).reset_index(drop=True)
df[target_col] = df[target_col].astype(int)

split_index = int(len(df) * 0.80)
df["Period"] = "Training"
df.loc[df.index >= split_index, "Period"] = "Testing"

summary = (
    df.groupby(["Order Region", "Period"], observed=True)
    .agg(
        Orders=(target_col, "size"),
        Late_Risk_Rate=(target_col, "mean"),
    )
    .reset_index()
)

summary["Late_Risk_Rate"] = (
    summary["Late_Risk_Rate"] * 100
).round(2)

wide = summary.pivot(
    index="Order Region",
    columns="Period",
    values=["Orders", "Late_Risk_Rate"],
)

wide.columns = [
    f"{metric}_{period}" for metric, period in wide.columns
]

for col in ["Orders_Training", "Orders_Testing"]:
    if col not in wide.columns:
        wide[col] = 0

for col in ["Late_Risk_Rate_Training", "Late_Risk_Rate_Testing"]:
    if col not in wide.columns:
        wide[col] = float("nan")

wide["Rate_Difference_pp"] = (
    wide["Late_Risk_Rate_Testing"]
    - wide["Late_Risk_Rate_Training"]
).round(2)

print("=" * 80)
print("SUPPLYGUARD - REGION AND TARGET COMPARISON")
print("=" * 80)

print("\nRegions with the largest absolute rate differences:")
print(
    wide.sort_values(
        "Rate_Difference_pp",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last",
    ).head(20).to_string()
)

print("\nRegions present in testing but absent from training:")
print(
    wide[
        (wide["Orders_Testing"].fillna(0) > 0)
        & (wide["Orders_Training"].fillna(0) == 0)
    ].to_string()
)

print("\nAnalysis complete. No model was trained or modified.")
