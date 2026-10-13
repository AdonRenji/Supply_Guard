
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATASET_PATH, encoding="latin1")

date_col = "order date (DateOrders)"
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

features = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Order Country",
    "Order State",
    "Order Region",
]

# Derive the same order-time features used in the experiments.
df["Order Month"] = df[date_col].dt.month
df["Order Day of Week"] = df[date_col].dt.dayofweek
df["Order Hour"] = df[date_col].dt.hour

features += [
    "Order Month",
    "Order Day of Week",
    "Order Hour",
]

# Use the same chronological 80/20 split.
split_index = int(len(df) * 0.80)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

print("=" * 70)
print("SUPPLYGUARD - FEATURE DISTRIBUTION OVER TIME")
print("=" * 70)

print("\nTraining period:")
print(train[date_col].min(), "to", train[date_col].max())
print("Rows:", len(train))

print("\nTesting period:")
print(test[date_col].min(), "to", test[date_col].max())
print("Rows:", len(test))

for feature in features:
    print("\n" + "-" * 70)
    print("FEATURE:", feature)

    if pd.api.types.is_numeric_dtype(df[feature]):
        print("Training statistics:")
        print(train[feature].describe().round(2).to_string())

        print("\nTesting statistics:")
        print(test[feature].describe().round(2).to_string())
    else:
        train_pct = (
            train[feature].value_counts(normalize=True) * 100
        ).round(2)

        test_pct = (
            test[feature].value_counts(normalize=True) * 100
        ).round(2)

        comparison = pd.concat(
            [
                train_pct.rename("Training %"),
                test_pct.rename("Testing %"),
            ],
            axis=1,
        ).fillna(0)

        comparison["Difference (percentage points)"] = (
            comparison["Testing %"] - comparison["Training %"]
        ).round(2)

        print(comparison.sort_values(
            "Difference (percentage points)",
            key=lambda s: s.abs(),
            ascending=False,
        ).head(15).to_string())

print("\nAnalysis complete. No model was trained or modified.")
