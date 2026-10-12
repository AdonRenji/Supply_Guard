import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATASET_PATH, encoding="latin1")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. CREATE DATE FEATURES
# ============================================================

df["DateOrders"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

df["Order Month"] = df["DateOrders"].dt.month
df["Order Day"] = df["DateOrders"].dt.day
df["Order Day of Week"] = df["DateOrders"].dt.dayofweek
df["Order Hour"] = df["DateOrders"].dt.hour

df["Is Weekend"] = (
    df["Order Day of Week"] >= 5
).astype(int)

df["Order Week of Year"] = (
    df["DateOrders"].dt.isocalendar().week.astype(int)
)

df["Order Day of Month"] = df["DateOrders"].dt.day

df["Is Month Start"] = (
    df["DateOrders"].dt.is_month_start
).astype(int)

df["Is Month End"] = (
    df["DateOrders"].dt.is_month_end
).astype(int)

df["Is Quarter Start"] = (
    df["DateOrders"].dt.is_quarter_start
).astype(int)

df["Is Quarter End"] = (
    df["DateOrders"].dt.is_quarter_end
).astype(int)

df["Is Business Hour"] = (
    (df["Order Hour"] >= 9) &
    (df["Order Hour"] <= 17)
).astype(int)

df["Is Night"] = (
    (df["Order Hour"] < 6) |
    (df["Order Hour"] >= 22)
).astype(int)


# ============================================================
# 3. CREATE INTERACTION FEATURE
# ============================================================

df["Interaction_ShippingMode_OrderHour"] = (
    df["Shipping Mode"].astype(str)
    + "_Hour_"
    + df["Order Hour"].astype(str)
)


# ============================================================
# 4. FINAL 31 FEATURES
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


X = df[features].copy()


# ============================================================
# 5. DEFINE CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "Shipping Mode",
    "Customer Country",
    "Category Name",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Customer Segment",
    "Department Name",
    "Interaction_ShippingMode_OrderHour",
    "Customer City",
    "Type",
    "Customer State"
]

numerical_features = [
    feature for feature in features
    if feature not in categorical_features
]


print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# ============================================================
# 6. CREATE SAME PREPROCESSING USED BY FINAL MODEL
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ============================================================
# 7. TRAIN/TEST SPLIT
# ============================================================

y = df["Late_delivery_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 8. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ============================================================

print("\nFitting One-Hot Encoder...")

X_train_encoded = preprocessor.fit_transform(X_train)

print("One-Hot Encoding completed!")

print(
    "Original features:",
    X_train.shape[1]
)

print(
    "Transformed features:",
    X_train_encoded.shape[1]
)


# ============================================================
# 9. GET ENCODED FEATURE NAMES
# ============================================================

encoded_feature_names = (
    preprocessor.get_feature_names_out()
)


# ============================================================
# 10. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# FILE 1
# COMPLETE LIST OF ENCODED FEATURE NAMES
# ============================================================

feature_names_df = pd.DataFrame({
    "Encoded_Feature": encoded_feature_names
})

feature_names_path = (
    "results/one_hot_encoding_values.csv"
)

feature_names_df.to_csv(
    feature_names_path,
    index=False
)

print(
    "\nFile 1 created:",
    feature_names_path
)


# ============================================================
# FILE 2
# SAMPLE OF ACTUAL ENCODED VALUES
# ============================================================

# Convert encoded NumPy array into DataFrame
encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=encoded_feature_names
)

# Take first 100 training records
encoded_sample = encoded_df.head(100)

sample_path = (
    "results/one_hot_encoding_sample.csv"
)

encoded_sample.to_csv(
    sample_path,
    index=False
)

print(
    "File 2 created:",
    sample_path
)


# ============================================================
# 11. CREATE A MORE READABLE MAPPING FILE
# ============================================================

mapping_rows = []

for category in categorical_features:

    # Get encoder belonging to categorical pipeline
    encoder = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    # Find index of this categorical feature
    feature_index = categorical_features.index(category)

    categories = encoder.categories_[feature_index]

    for value in categories:

        mapping_rows.append({
            "Original_Feature": category,
            "Original_Value": value,
            "Encoded_Column":
                f"cat__{category}_{value}"
        })


mapping_df = pd.DataFrame(mapping_rows)

mapping_path = (
    "results/one_hot_encoding_mapping.csv"
)

mapping_df.to_csv(
    mapping_path,
    index=False
)

print(
    "Mapping file created:",
    mapping_path
)


# ============================================================
# 12. FINAL INFORMATION
# ============================================================

print("\n==========================================")
print("ONE-HOT ENCODING FILES CREATED")
print("==========================================")

print("\n1. one_hot_encoding_values.csv")
print("   Contains all transformed feature names.")

print("\n2. one_hot_encoding_sample.csv")
print("   Contains actual encoded 0/1 values")
print("   for the first 100 training records.")

print("\n3. one_hot_encoding_mapping.csv")
print("   Shows original categorical values")
print("   and their corresponding encoded columns.")

print("\nTotal transformed features:",
      len(encoded_feature_names))

print("\nCompleted successfully!")